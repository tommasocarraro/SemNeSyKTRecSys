import ast
import os
import time
from joblib import Parallel, delayed
import json
from multiprocessing import Manager
import csv
import pandas as pd
import requests
import re
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import wayback
import sys
from SPARQLWrapper import SPARQLWrapper, JSON
import logging
import math
from tqdm import tqdm
import subprocess
import sqlite3
import pprint


def create_asin_metadata_json(metadata):
    """
    This function takes as input the raw JSON file containing Amazon product metadata and produces a filtered version of
    the same file, where for each ASIN we store information about just the title. The title of the Amazon products is
    used in this project to match the Amazon items into the Wikidata ontology. Wikidata provides a special search
    service for which it is enough to provide the title of an item to get a list of possible Wikidata candidates.

    The new JSON file is saved into the /processed folder and it is called metadata.json. It is the same as
    /raw/metadata.json but containing only the titles of the items.

    :param metadata: path of raw JSON file containing Amazon product metadata
    """
    manager = Manager()
    md_dict = manager.dict()

    def process_line(line, md_dict):
        """
        Function used to process one line of the big Amazon metadata JSON file. It processes the line and saves only
        the relevant information (title in this case) into the given dictionary. The dictionary is indexed by the
        ASIN of the products.

        When no information is available on the metadata file, it keeps track of it for then using scraping to scrape
        relevant information from the web. A "no-title" string is put in place of the title for each of these cases.

        :param line: line of JSON file containing metadata
        :param md_dict: dictionary containing only relevant metadata (title for each ASIN)
        """
        row_dict = ast.literal_eval(line)
        try:
            md_dict[row_dict["asin"]] = row_dict["title"]
        except Exception as e:
            # when an exception occurs, it means the title is not available in the metadata file
            print(e)
            md_dict[row_dict["asin"]] = "no-title"

    # process file and save relevant data
    with open(metadata) as f:
        Parallel(n_jobs=os.cpu_count())(delayed(process_line)(line, md_dict) for line in f)

    # save the new file
    with open('./data/processed/%s' % (metadata.split("/")[-1]), 'w', encoding='utf-8') as f:
        json.dump(md_dict.copy(), f, ensure_ascii=False, indent=4)


def filter_metadata(metadata, rating_files):
    """
    This function filters the given metadata file by only keeping the items included in the rating files provided in
    input. The Amazon dataset comprises plentiful of rating files. In this project, we are interested in movies, music,
    and books. For this reason, we are interested in filtering out all the remaining items from the metadata file.
    This will make the next steps more efficient.

    The filtering process just creates a new metadata file where only the items included in the provided rating files
    are kept.

    It creates a new metadata file in the same location of metadata (/processed), called filtered-metadata.json.

    :param metadata: path to processed metadata file (containing only title for each ASIN)
    :param rating_files: list of rating file names. Provide only the name (e.g., reviews_Movies_and_TV_5). No extension
    or path is required. The path is supposed to be /processed.
    """
    # create a list of interested ASINs
    desired_asins = []
    for dataset in rating_files:
        data = pd.read_csv("./data/processed/%s.csv" % (dataset,))
        desired_asins.extend(data["itemId"].unique())

    with open(metadata) as json_file:
        m_data = json.load(json_file)

    # get the metadata for the desired ASINs
    new_file_dict = {k: m_data[k] for k in desired_asins}

    # create the new metadata file with only the relevant items
    with open('./data/processed/filtered-%s' % (metadata.split("/")[-1]), 'w', encoding='utf-8') as f:
        json.dump(new_file_dict, f, ensure_ascii=False, indent=4)


def scrape_title(asins, n_cores, batch_size, save_tmp=True):
    """
    This function takes as input a list of Amazon ASINs and performs http requests with Selenium to get the title
    corresponding to the ASIN from the Amazon website. Sometimes, the script is detected by Amazon and a captcha is
    displayed in the page. In such cases, the script keeps track of the bot detection. This allows to execute the script
    again only for those items that have been bot-detected during the first run.

    :param asins: list of ASINs for which the title has to be retrieved
    :param n_cores: number of cores to be used for scraping
    :param batch_size: number of ASINs to be processed in each batch of parallel execution
    :param save_tmp: whether temporary retrieved title JSON files have to be saved once the batch is finished. This
    allows saving everything in cases in which the script is interrupted by external forces.
    :return: new dictionary containing key-value pairs with scraped ASIN-title
    """
    # get number of batches for printing information
    n_batches = int(len(asins) / batch_size)
    # define dictionary suitable for parallel storing of information
    manager = Manager()
    title_dict = manager.dict()
    # Set up the Chrome options for a headless browser
    chrome_options = Options()
    chrome_options.add_argument('--disable-gpu')  # Disable GPU to avoid issues in headless mode
    chrome_options.add_argument('--window-size=1920x1080')  # Set a window size to avoid responsive design

    def batch_request(batch_idx, asins, title_dict):
        """
        This function performs a batch of HTTP requests using Selenium.

        :param batch_idx: index of the current batch
        :param asins: list of ASINs of the products for which the title has to be retrieved in this batch
        :param title_dict: dictionary for parallel storing of the retrieved data
        """
        # check if this batch has been already processed in another execution
        # if the path does not exist, we process the batch
        tmp_path = "./data/processed/metadata-batch-%s" % (batch_idx,)
        if not os.path.exists(tmp_path):
            # define dictionary for saving batch data
            batch_dict = {}
            # create the URLs for scraping
            urls = [f'https://www.amazon.com/dp/{asin}' for asin in asins]
            # Set up the Chrome driver for the current batch
            chrome_service = ChromeService(executable_path='./chromedriver')
            driver = webdriver.Chrome(service=chrome_service, options=chrome_options)
            # start the scraping loop
            for counter, url in enumerate(urls):
                asin = url.split("/")[-1]
                try:
                    # Load the Amazon product page
                    driver.get(url)
                    # get the page
                    page = driver.page_source
                    # Parse the HTML content of the page
                    soup = BeautifulSoup(page, 'html.parser')
                    # Find the product title
                    title_element = soup.find('span', {'id': 'productTitle'})
                    if title_element:
                        # the title has been found and we save it in the dictionary
                        print_str = title_element.text.strip()
                        batch_dict[asin] = title_element.text.strip()
                    else:
                        # the title has not been found
                        # check if it is due to a 404 error
                        error = soup.find('img', {'alt': "Sorry! We couldn't find that page. "
                                                         "Try searching or go to Amazon's home page."})
                        if error:
                            # if it is due to 404 error, keeps track of it
                            # items not found will be processed in another scraping loop that uses Wayback Machine
                            print_str = "404 error"
                            batch_dict[asin] = "404-error"
                        else:
                            # if it is not a 404 error, the bot has been detected
                            print_str = "Bot detected - url %s" % (url,)
                            # it could be because of a captcha from Amazon or also because there is not productTitle
                            # but another ID
                            # items bot-detected will be processed in another scraping loop that tries again
                            # if the problem is related to the DOM, the web page has to be investigated
                            batch_dict[asin] = "captcha-or-DOM"
                except Exception as e:
                    print(e)
                    print_str = "unknown error"
                    # if an exception is thrown by the system, I am interested in knowing which ASIN caused that, so
                    # I keep track of the exception in the dictionary
                    batch_dict[asin] = "exception-error"
                print("batch %d / %d - asin %d / %d - %s" % (batch_idx, n_batches, counter, len(asins), print_str))
            # after each batch, the resources allocated by Selenium have to be realised
            driver.quit()
            if save_tmp:
                # save a json file dedicated to this specific batch
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    json.dump(batch_dict, f, ensure_ascii=False, indent=4)
        else:
            # load the file and update the parallel dict
            with open(tmp_path) as json_file:
                batch_dict = json.load(json_file)
        # update parallel dict
        title_dict.update(batch_dict)

    # parallel scraping -> asins are subdivided into batches and the batches are run in parallel
    Parallel(n_jobs=n_cores)(delayed(batch_request)(batch_idx, a, title_dict)
                             for batch_idx, a in
                             enumerate([asins[i:(i + batch_size if i + batch_size < len(asins) else len(asins))]
                                        for i in range(0, len(asins), batch_size)]))
    return title_dict.copy()


def scrape_title_wayback(asins, batch_size=20, save_tmp=True, delay=60):
    """
    This function takes as input a list of Amazon ASINs and performs http requests with an unofficial Wayback API to get
    the title of the ASIN from the Wayback Machine website. This is used for the ASINs that during the first scraping
    job obtained a 404 error on Amazon. This is an attempt of still retrieve the title despite the official page of
    the product not existing anymore. Note that parallel execution is not possible with Wayback Machine as they are
    blocking for one minute every 20 HTTP requests.

    If the script is able to find a title given the ASIN, the title is saved inside a dictionary. If a captcha or 404
    error is detected even in the Wayback Machine, the script keeps track of it. Finally, if there is a DOM problem,
    namely the title cannot be found due to an ID not found in the DOM of the web page, the script keeps track of it
    and we suggest the user to manually put the match inside the final JSON file.
    and we suggest the user to manually put the match inside the final JSON file.

    :param asins: list of ASINs for which the title has to be retrieved
    :param batch_size: number of ASINs to be processed in each batch. Default to 20 as adding ASINs to the batch will
    cause a block from the Wayback API
    :param save_tmp: whether temporary retrieved title JSON files have to be saved once the batch is finished
    :param delay: number of seconds to wait between one batch and another. Default to 60 seconds as we need to wait
    one minute every 20 HTTP request to avoid being blocked
    :return: new dictionary containing key-value pairs with ASIN-title
    """
    # get number of batches for printing information
    n_batches = int(len(asins) / batch_size)
    # define dictionary suitable for storing of information
    title_dict = dict()

    def batch_request(batch_idx, asins, title_dict):
        """
        This function performs a batch of HTTP requests using an unofficial Wayback API.

        :param batch_idx: index of the current batch
        :param asins: list of ASINs of the products for which the title has to be retrieved in this batch
        :param title_dict: dictionary for storing of the retrieved data
        """
        # check if this batch has been already processed in another execution
        # if the path does not exist, we process the batch
        tmp_path = "./data/processed/metadata-batch-%s" % (batch_idx,)
        if not os.path.exists(tmp_path):
            # define dictionary for saving batch data
            batch_dict = {}
            # define the Amazon URLs that have to be searched in the Wayback Machine
            urls = [f'https://www.amazon.com/dp/{asin}' for asin in asins]
            # define wayback API
            client = wayback.WaybackClient()
            # start the scraping loop
            for counter, url in enumerate(urls):
                print_str = ""
                asin = url.split("/")[-1]
                found = False
                # iterate over all the snapshots and block when you find the first that has some useful content,
                # for example, that it is not a captcha and has a meaningful title available
                for snapshot in client.search(url):
                    try:
                        # get the content of the snapshot
                        page = client.get_memento(snapshot).content
                        # if we arrive at this point without a MementoException, it means the link has been found
                        # in Wayback Machine
                        found = True
                        # parse the saved page content
                        soup = BeautifulSoup(page, 'html.parser')  # html.parser
                        # check if the page is a captcha page and discard it from the search
                        if soup.find("h4") is not None and soup.find("h4").get_text() == ("Enter the characters you "
                                                                                          "see below"):
                            batch_dict[asin] = "captcha"
                            print_str = "captcha problem - ASIN %s" % (url,)
                            # move to next snapshot in the for loop
                            continue
                        # check if it is a saved page with 404 error
                        if ((soup.find("b", {"class": "h1"}) is not None and
                             "Looking for something?" in soup.find("b", {"class": "h1"}).get_text()) or
                                soup.find('img', {'alt': "Sorry! We couldn't find that page. "
                                                         "Try searching or go to Amazon's home page."}) is not None):
                            batch_dict[asin] = "404-error"
                            print_str = "404 error - ASIN %s" % (url,)
                            # move to next snapshot in the for loop
                            continue
                        # if it is not a captcha page or 404 page, find this specific IDs (in order) while
                        # parsing the web page
                        id_alternatives = ["btAsinTitle", "productTitle", "ebooksProductTitle"]
                        title_element = soup.find('span', {'id': id_alternatives[0]})
                        i = 1
                        while title_element is None and i < len(id_alternatives):
                            title_element = soup.find('span', {'id': id_alternatives[i]})
                            i += 1
                        if title_element is not None:
                            batch_dict[asin] = title_element.text.strip()
                            print_str = title_element.text.strip()
                            # one has been found, no need to continue the search
                            break
                        else:
                            # if none of the IDs has been found, the page is very old and we need to search for
                            # a "b" with class "sans"
                            title_element = soup.find('b', {'class': "sans"})
                            if title_element is not None:
                                batch_dict[asin] = title_element.text.strip()
                                print_str = title_element.text.strip()
                                # one has been found, no need to continue the search
                                break
                            else:
                                # if none of the IDs neither the b has been found, there is a DOM problem, meaning
                                # there could be another possible ID or soup is not working as expected
                                batch_dict[asin] = "DOM"
                                print_str = "DOM problem - ASIN %s" % (url,)
                    except Exception as e:
                        # this occurs when there is a MementoException -> we are interested in knowing the exception
                        print_str = e
                if not found:
                    # if we enter here, it means that no web page has been saved in Wayback Machine for the given URL
                    print_str = "404 error - ASIN %s" % (url,)
                    batch_dict[asin] = "404-error"
                print("batch %d / %d - asin %d / %d - %s" % (batch_idx, n_batches, counter, len(asins), print_str))
            if save_tmp:
                # save a json file dedicated to this specific batch
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    json.dump(batch_dict, f, ensure_ascii=False, indent=4)
            # wait some seconds between a batch and the other
            time.sleep(delay)
        else:
            # load the file and update the parallel dict
            with open(tmp_path) as json_file:
                batch_dict = json.load(json_file)
        # update parallel dict
        title_dict.update(batch_dict)

    # begin scraping loop
    for batch_idx, batch_asins in enumerate([asins[i:(i + batch_size if i + batch_size < len(asins) else len(asins))]
                                             for i in range(0, len(asins), batch_size)]):
        batch_request(batch_idx, batch_asins, title_dict)
    return title_dict.copy()


def scrape_title_captcha(asins, batch_size=100, save_tmp=True, delay=180):
    """
    This function takes as input a list of Amazon ASINs and performs http requests to the Rocket Source knowledge base
    to get the title of the ASIN. This is used for the ASINs that during the second scraping job (Wayback machine)
    obtained a 404 error on the Wayback machine. This is an attempt of still retrieve the title despite the official
    page of the product not existing anymore even on the Wayback Machine.

    If the script is able to find a title given the ASIN, the title is saved inside a dictionary. If the ASIN is not
    included in the database, the script keeps track of it.

    :param asins: list of ASINs for which the title has to be retrieved
    :param batch_size: number of ASINs to be processed in each batch
    :param save_tmp: whether temporary retrieved title JSON files have to be saved once the batch is finished
    :param delay: number of seconds to wait for solving captchas. It is possible the tool requires a lot of time for
    solving a CAPTCHA, depending on how much challenging it is.
    :return: new dictionary containing key-value pairs with ASIN-title
    """
    # get number of batches for printing information
    n_batches = int(len(asins) / batch_size)
    # define dictionary suitable for parallel storing of information
    manager = Manager()
    title_dict = manager.dict()
    # Set up the Chrome options for a headless browser
    options = Options()
    options.add_argument('--disable-gpu')  # Disable GPU to avoid issues in headless mode
    options.add_argument('--window-size=1920x1080')  # Set a window size to avoid responsive design
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-infobars')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option('excludeSwitches', ['enable-automation'])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_extension("./nocaptcha/noCaptchaAi-chrome-v1.3.crx")
    # url for configuring the extension for captchas
    url_api = "https://newconfig.nocaptchaai.com/?APIKEY=bmxitalia-5f6ff273-aa2e-2ffb-fff7-c0388051fe71&PLANTYPE=pro&customEndpoint=&hCaptchaEnabled=true&reCaptchaEnabled=true&dataDomeEnabled=true&ocrEnabled=true&ocrToastEnabled=true&extensionEnabled=true&logsEnabled=false&fastAnimationMode=true&debugMode=false&hCaptchaAutoOpen=true&hCaptchaAutoSolve=true&hCaptchaAlwaysSolve=true&englishLanguage=true&hCaptchaGridSolveTime=7&hCaptchaMultiSolveTime=5&hCaptchaBoundingBoxSolveTime=5&reCaptchaAutoOpen=true&reCaptchaAutoSolve=true&reCaptchaAlwaysSolve=true&reCaptchaClickDelay=400&reCaptchaSubmitDelay=1&reCaptchaSolveType=null"
    # url of the webpage of Rocket Source knowledge base
    url = "https://www.rocketsource.io/asin-to-ean"

    def batch_request(batch_idx, asins, title_dict):
        """
        This function performs a batch of HTTP requests to the Rocket Source knowledge base.

        :param batch_idx: index of the current batch
        :param asins: list of ASINs of the products for which the title has to be retrieved in this batch
        :param title_dict: dictionary for storing of the retrieved data
        """
        # check if this batch has been already processed in another execution
        # if the path does not exist, we process the batch
        tmp_path = "./data/processed/metadata-batch-%s" % (batch_idx,)
        if not os.path.exists(tmp_path):
            # define dictionary for saving batch data
            batch_dict = {}
            # Set up the Chrome driver for the current batch
            chrome_service = ChromeService(executable_path='./chromedriver')
            driver = webdriver.Chrome(service=chrome_service, options=options)
            # config the extension
            driver.get(url_api)
            # open the webpage
            driver.get(url)
            # Find the input element where the ASIN has to be inserted
            input_element = driver.find_element(By.TAG_NAME, "input")
            # start the scraping loop
            for counter, asin in enumerate(asins):
                print_str = ""
                # Input the ASIN into the input element
                input_element.clear()
                input_element.send_keys(asin)
                # wait until the button becomes clickable - we have to wait for the captcha to be solved
                try:
                    wait = WebDriverWait(driver, delay)
                    button = wait.until(EC.presence_of_element_located((By.XPATH,
                                                                        '//*[@id="__next"]/div/div[2]/div/div/div[3]/div[1]/div[1]/div[3]/div[2]/div[contains(@class, "cursor-pointer")]')))
                except Exception as e:
                    # if we enter here, it means that the timeout run out of time and the captcha has not be solved by
                    # the tool
                    print_str = "captcha not solved - ASIN %s" % (asin,)
                    batch_dict[asin] = "captcha"
                    print("batch %d / %d - asin %d / %d - %s" % (batch_idx, n_batches, counter, len(asins), print_str))
                    # continue to the next ASIN
                    continue
                # if we arrive here, the captcha has been solved
                # click on the button to generate the title on the Rocket Source site
                button.click()
                try:
                    # wait for the title to appear
                    wait = WebDriverWait(driver, 10)
                    title = wait.until(EC.presence_of_element_located(
                        (By.XPATH, '//*[@id="__next"]/div/div[2]/div/div/div[3]/div[1]/div[2]/b')))
                except Exception as e:
                    print_str = "not found - ASIN %s" % (asin,)
                    print("batch %d / %d - asin %d / %d - %s" % (batch_idx, n_batches, counter, len(asins), print_str))
                    batch_dict[asin] = "404-error"
                    continue
                # if we arrive here, the title has been correctly displayed
                print_str = "%s - %s" % (asin, title.text)
                print("batch %d / %d - asin %d / %d - %s" % (batch_idx, n_batches, counter, len(asins), print_str))
                batch_dict[asin] = title.text
            # Close the browser window
            driver.quit()
            if save_tmp:
                # save a json file dedicated to this specific batch
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    json.dump(batch_dict, f, ensure_ascii=False, indent=4)
        else:
            # load the file and update the parallel dict
            with open(tmp_path) as json_file:
                batch_dict = json.load(json_file)
        # update parallel dict
        title_dict.update(batch_dict)

    # parallel scraping -> asins are subdivided into batches and the batches are run in parallel
    Parallel(n_jobs=os.cpu_count())(delayed(batch_request)(batch_idx, a, title_dict)
                                    for batch_idx, a in
                                    enumerate([asins[i:(i + batch_size if i + batch_size < len(asins) else len(asins))]
                                               for i in range(0, len(asins), batch_size)]))
    return title_dict.copy()


def scrape_title_google_search(asins, batch_size=100, save_tmp=True, just_first=True):
    """
    This function takes as input a list of Amazon ASINs and performs http requests to the Google Search
    to get the title of the ASIN. It simply searches on the Google Search text area and iterates over the results to get
    the title of the searched ASIN. This is used for the ASINs that during the second scraping job (Wayback machine)
    obtained a 404 error on the Wayback machine. This is an attempt of still retrieve the title despite the official
    page of the product not existing anymore even on the Wayback Machine.

    If the script is able to find a title given the ASIN, the title is saved inside a dictionary. If the search does not
    produce any results, the script keeps track of it.

    :param asins: list of ASINs for which the title has to be retrieved
    :param batch_size: number of ASINs to be processed in each batch
    :param save_tmp: whether temporary retrieved title JSON files have to be saved once the batch is finished
    :param just_first: whether to save just the title of the first link of the search or all the titles that are found
    by Google for this specific search
    :return: new dictionary containing key-value pairs with ASIN-title
    """
    # define dictionary suitable for parallel storing of information
    manager = Manager()
    title_dict = manager.dict()
    # Set up the Chrome options for a headless browser
    options = Options()
    options.add_argument('--disable-gpu')  # Disable GPU to avoid issues in headless mode
    options.add_argument('--window-size=1920x1080')  # Set a window size to avoid responsive design
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-infobars')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')

    def batch_request(batch_idx, asins, title_dict):
        """
        This function performs a batch of HTTP requests to the Google Search.

        :param batch_idx: index of the current batch
        :param asins: list of ASINs of the products for which the title has to be retrieved in this batch
        :param title_dict: dictionary for storing of the retrieved data
        """
        # check if this batch has been already processed in another execution
        # if the path does not exist, we process the batch
        tmp_path = "./data/processed/metadata-batch-%s" % (batch_idx,)
        if not os.path.exists(tmp_path):
            # define dictionary for saving batch data
            batch_dict = {}
            # Set up the Chrome driver for the current batch
            chrome_service = ChromeService(executable_path='./chromedriver')
            driver = webdriver.Chrome(service=chrome_service, options=options)
            # start the scraping loop
            for counter, asin in enumerate(asins):
                print_str = ""
                # search for ASIN in Google Search
                url = "http://www.google.com/search?as_q=" + asin
                driver.get(url)
                # get page source
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                # check if Google changed my search
                a = soup.find('a', class_='spell_orig')
                if a is not None:
                    # get the link and go to the correct search
                    url = a.get('href')
                    driver.get('http://www.google.com' + url)
                    # get new page source
                    soup = BeautifulSoup(driver.page_source, 'html.parser')
                # iterate over the links of the page
                divs = soup.find_all('div', class_="yuRUbf")
                link_list = []
                for div in divs:
                    link_list.append(div.h3.text)
                if link_list and just_first:
                    link_list = link_list[0]
                print("%s - %s" % (asin, link_list))
                if not link_list:
                    link_list = "not-title"
                batch_dict[asin] = link_list
            # Close the browser window
            driver.quit()
            if save_tmp:
                # save a json file dedicated to this specific batch
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    json.dump(batch_dict, f, ensure_ascii=False, indent=4)
        else:
            # load the file and update the parallel dict
            with open(tmp_path) as json_file:
                batch_dict = json.load(json_file)
        # update parallel dict
        title_dict.update(batch_dict)

    # parallel scraping -> asins are subdivided into batches and the batches are run in parallel
    Parallel(n_jobs=os.cpu_count())(delayed(batch_request)(batch_idx, a, title_dict)
                                    for batch_idx, a in
                                    enumerate([asins[i:(i + batch_size if i + batch_size < len(asins) else len(asins))]
                                               for i in range(0, len(asins), batch_size)]))
    return title_dict.copy()


def metadata_scraping(metadata, n_cores=1, motivation="no-title", save_tmp=True, batch_size=100, mode=None):
    """
    This function takes as input a metadata file with some missing titles and uses web scraping to retrieve these
    titles from the Amazon website. It is possible to specify wayback=True to scrape titles from the Wayback Machine.
    This process is very inefficient and is suggested only for items that produced a 404 error on Amazon.

    At the end, a new metadata file with complete information is generated. This metadata file is called
    complete-<input-file-name>.json.

    :param metadata: file containing metadata for the items
    :param n_cores: number of cores to be used for scraping. This has no effect when wayback=True.
    :param motivation: motivation for which the title has to be scraped. It could be:
        - no-title (default): no-title means that the metadata is missing the title for the item. Put
        motivation="no-title" only for the first scraping job, to get the title using the Amazon website
        - captcha-or-DOM: during the first scraping job, the title retrieval for this item failed due to incorrect DOM
        or bot detection from Amazon. If this motivation is selected, the script will take all the ASINs related to this
        error code and will perform another scraping loop on them
        - 404-error: during the first scraping job, the title retrieval failed due to a 404 not found error, meaning the
        ASIN is not on Amazon anymore. If this motivation is selected, the script will take all the ASINs related to
        this error code and will perform a scraping loop using Wayback Machine. Remember to put wayback=True
        - exception-error: in the first scraping job, the title retrieval failed due to an exception
    :param save_tmp: whether temporary retrieved title JSON files have to be saved once the batch is finished
    :param batch_size: number of ASINs that have to be processed for each batch. Keep this number under 100 to avoid
    disk memory problems due to temporary files memorization by Selenium. Keep this number equal to 20 when wayback=True
    :param mode: - "wayback" uses the wayback machine API (for items not found after first scraping loop)
                 - "standard" uses the standard one (for first scraping loop)
                 - "captcha" uses the version that scrapes titles from the Rocket Source knowledge base (to be used
                 after the wayback mode. If even with the Wayback Machine is not possible to get the titles, it is
                 possible to get them from this knowledge base. It is the slowest mode as it requires solving difficult
                 captchas automatically)
                 - "search" uses the Google Search to get the titles corresponding to the given ASINs
    """
    with open(metadata) as json_file:
        m_data = json.load(json_file)
    # take the ASINs for the products that have a missing title in the metadata file
    # only the items with the selected motivation will be picked
    no_titles = [k for k, v in m_data.items() if v == motivation]
    # update the metadata with the scraped titles
    if mode is None or mode == "standard":
        updated_dict = scrape_title(no_titles, n_cores, batch_size=batch_size, save_tmp=save_tmp)
    elif mode == "wayback":
        updated_dict = scrape_title_wayback(no_titles, batch_size=batch_size, save_tmp=save_tmp)
    elif mode == "captcha":
        updated_dict = scrape_title_captcha(no_titles, batch_size=batch_size, save_tmp=save_tmp)
    else:
        updated_dict = scrape_title_google_search(no_titles, batch_size=batch_size, save_tmp=save_tmp)
    # update of the metadata
    m_data.update(updated_dict)
    # generate the new and complete metadata file
    with open('./data/processed/complete-%s' % (metadata.split("/")[-1]), 'w', encoding='utf-8') as f:
        json.dump(m_data, f, ensure_ascii=False, indent=4)


def metadata_stats(metadata, errors, save_asins=True):
    """
    This function produces some statistics for the provided metadata file. The statistics include the number of items
    with a missing title due to a 404 error, a bot detection or DOM error, or an unknown error due to an exception in
    the scraping procedure. The parameter errors allows to define which statistics to include in the output.
    For each of these statistics, the output will contain the set of ASINs corresponding to each error, if save_asins
    is set to True. The output will also contain the percentage of items included in each statistics given the total
    number of items in the given file.

    :param metadata: path to the metadata file for which the statistics have to be generated
    :param errors: list of strings containing the name of the errors that have to be included in the statistics. The
    script will search for these specific names in the values of the metadata, for each of the ASINs
    :param save_asins: whether to save the set of the ASINs for each statistic or just produce the statistic
    """
    errors = {e: {"counter": 0, "asins": []} for e in errors}
    with open(metadata) as json_file:
        m_data = json.load(json_file)
    for asin, title in m_data.items():
        if title in errors:
            errors[title]["counter"] += 1
            if save_asins:
                errors[title]["asins"].append(asin)
        else:
            if "matched" in errors:
                errors["matched"]["counter"] += 1
            else:
                errors["matched"] = {"counter": 1}
    # compute percentages
    total = sum([errors[e]["counter"] for e in errors])
    for e in errors:
        errors[e]["percentage"] = errors[e]["counter"] / total * 100
    print(errors)


def metadata_cleaning(metadata):
    """
    This function takes as input a metadata file after all the scraping jobs have been completed (only 404 error and
    matched titles should remain) and remove all the ASINs with 404 error.

    A new file with only ASIN-title pairs is created.

    :param metadata: metadata file that has to be cleaned
    """
    with open(metadata) as json_file:
        m_data = json.load(json_file)
    m_data = {k: v for k, v in m_data.items() if v != "404-error"}
    with open("./data/processed/final-metadata.json", 'w', encoding='utf-8') as f:
        json.dump(m_data, f, ensure_ascii=False, indent=4)


def create_pandas_dataset(data):
    """
    This function takes as input a JSON file containing Amazon reviews and produces a CSV file. Each record has
    user ID, item ID, rating, timestamp. In other words, it creates a rating file based on the given review file.

    This is a way of filtering out data that is not useful for this projects, for example, the review text. We just
    need ratings on items and their metadata to link the items in the ontology.

    The new file is saved into the /processed folder, with the same name as the one of the original file.

    :param data: path to the JSON file containing Amazon reviews
    """
    # defining dictionary for parallel storing of ratings
    manager = Manager()
    csv_list = manager.list()

    def process_line(line, rating_list):
        """
        Function used to process one line of the Amazon review file. It processes the line and saves only
        the relevant information into the given dictionary (in this case, user and item IDs, rating, and timestamp).

        :param line: line of JSON file containing ratings
        :param rating_list: list containing ratings to be saved in the CSV file. Each rating is a dict.
        """
        # read line
        row_dict = ast.literal_eval(line)
        try:
            # save relevant data
            rating_list.append({"userId": row_dict["reviewerID"], "itemId": row_dict["asin"],
                                "rating": row_dict["overall"], "timestamp": row_dict["unixReviewTime"]})
        except Exception as e:
            print(e)

    # parallel processing of the rating file
    with open(data) as f:
        Parallel(n_jobs=os.cpu_count())(delayed(process_line)(line, csv_list) for line in tqdm(f))

    # save the stored information in a new CSV rating file
    file_path = data.split("/")
    file_path[-2] = "processed"
    file_path[-1] = file_path[-1][:-4] + "csv"
    file_path = "/".join(file_path)
    with open(file_path, 'w', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["userId", "itemId", "rating", "timestamp"])
        writer.writeheader()
        writer.writerows(csv_list)


def entity_linker_api(amazon_ratings, use_dump=True):
    """
    This function uses the Wikidata API to get the ID of wikidata items corresponding to the Amazon items. The API is
    accessed using HTTP requests. It takes as input a CSV file containing ratings on Amazon items. The items are
    referred trough an ID, that is then used to take the corresponding title in the final metadata file. Then, the
    title is used for producing the Wikidata query. Depending on the
    rating file (movies, music, or books), the function uses the correct JSON dump file to check the validity of the
    found match. For example, if the query for the title Revolver produces the entity corresponding to the gun instead
    of the Beatles' album, the match is not created and discarded. This happens until the retrieved Wikidata item is
    of the correct category. If a match of the correct category is found, it is saved, otherwise no match is created.

    It creates a JSON file containing the mapping from Amazon ASIN to Wikidata ID.

    :param amazon_ratings: CSV file containing ratings on Amazon items
    :param use_dump: whether a JSON dump has to be used to check that the retrieved ID is of the correct category,
    this relies in the completeness of the dump (it is not easy to produce a complete dump with the Wikidata
    Query Service)
    """
    # read rating file
    data = pd.read_csv(amazon_ratings)
    # get item IDs
    item_ids = data['itemId'].unique()
    # read metadata
    with open("./data/processed/final_metadata.json") as json_file:
        m_data = json.load(json_file)
    # get the correct wikidata dump based on the given rating file
    wikidata_dump = ""
    if "Movies" in amazon_ratings:
        wikidata_dump = "./data/processed/wikidata-movies.json"
    if "Books" in amazon_ratings:
        wikidata_dump = ".data/processed/wikidata-books.json"
    if "CD" in amazon_ratings:
        wikidata_dump = "./data/processed/wikidata-music.json"
    # read the dump
    with open(wikidata_dump) as json_file:
        wikidata_dump = json.load(json_file)
    # link to API
    wikidata_api_url = "https://www.wikidata.org/w/api.php"

    def entity_link(asin):
        """
        It performs an HTTP request on the Wikidata API to get the Wikidata ID of the given item (referred by ASIN).

        :param asin: asin of the Amazon item for which the ID is requested
        """
        # get the title of the item without brackets (the brackets make the search more difficult for the Wikidata API)
        amazon_title = re.sub("[\(\[].*?[\)\]]", "", m_data[asin]["title"])
        # Define parameters for the Wikidata API search
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "search": amazon_title,
            "type": "item"
        }
        # Make the API request
        response = requests.get(wikidata_api_url, params=params)
        data = response.json()
        # Extract the Wikidata ID from the response
        if "search" in data and data["search"]:
            if use_dump:
                # if the dump has to be used, we need to check the Wikidata ID is included in the corresponding dump
                for item in data["search"]:
                    if item["id"] in wikidata_dump:
                        match_dict[asin] = item["id"]
                        break
            else:
                match_dict[asin] = data["search"][0]["id"]
        else:
            match_dict[asin] = ""

    # launch multiple queries in parallel and save the matches in the dictionary
    manager = Manager()
    match_dict = manager.dict()
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        executor.map(entity_link, item_ids)
    match_dict = match_dict.copy()
    with open('./data/processed/mapping-%s' % (amazon_ratings.split("/")[-1],), 'w', encoding='utf-8') as f:
        json.dump(match_dict, f, ensure_ascii=False, indent=4)


def entity_linker_api_query(amazon_ratings, use_dump=True, retry=False, retry_reason=None):
    """
    This function uses the Wikidata API (action=query) to get the ID of wikidata items corresponding to the Amazon
    items. The API is
    accessed using HTTP requests. It takes as input a CSV file containing ratings on Amazon items. The items are
    referred through an ID, that is then used to take the corresponding title in the final metadata file (all ASINs
    have a matched title in this file). Then, the title is used for producing the Wikidata query. Depending on the
    rating file (movies, music, or books), the function uses the correct JSON dump file to check the validity of the
    found match. For example, if the query for the title Revolver produces the entity corresponding to the gun instead
    of the Beatles' album, the match is not created and discarded. This happens until the retrieved Wikidata item is
    of the correct category. If a match of the correct category is found, it is saved, otherwise no match is created.

    It creates a JSON file containing the mapping from Amazon ASIN to Wikidata ID.

    :param amazon_ratings: CSV file containing ratings on Amazon items
    :param use_dump: whether a JSON dump has to be used to check that the retrieved ID is of the correct category,
    this relies on in the completeness of the dump (it is not easy to produce a complete dump with the Wikidata
    Query Service)
    :param retry: valid from second execution. If this is set to true, the script will retry to find matches for all
    the items that during the first search got a not-in-dump error. This is useful if one wants to update the dump file
    and retry the search.
    :param retry_reason: string indicating for which items the search has to be computed again. For example, indicate
    "exception" for the items that gave an exception during the first mapping loop. The procedure will repeat the loop
    only for these items.
    """
    # read data
    data = pd.read_csv(amazon_ratings)
    # get item IDs
    item_ids = data['itemId'].unique()
    # read metadata
    with open("./data/processed/final-metadata.json") as json_file:
        m_data = json.load(json_file)
    # get wikidata dump
    wikidata_dump, additional_dump = "", ""
    if "Movies" in amazon_ratings:
        wikidata_dump = "./data/processed/wikidata-movies.json"
    if "Books" in amazon_ratings:
        wikidata_dump = "./data/processed/wikidata-books.json"
    if "CD" in amazon_ratings:
        wikidata_dump = "./data/processed/wikidata-music.json"
        # this additional dump is just the same as the movie dump. It is used only for CDs and Vinyls, as some DVDs are
        # erroneously included in the CDs_and_Vinyls dataset and treated as CDs
        additional_dump = "./data/processed/wikidata-movies.json"
    # load the dump file
    if "music" in wikidata_dump:
        with open(additional_dump) as json_file:
            additional_dump = json.load(json_file)
    with open(wikidata_dump) as json_file:
        wikidata_dump = json.load(json_file)
    # link to API
    wikidata_api_url = "https://www.wikidata.org/w/api.php"
    # check if a mapping file for the given rating file already exists
    temp_dict = {}
    map_path = ("./data/processed/mapping-%s" % (amazon_ratings.split("/")[-1],)).replace("csv", "json")
    if os.path.exists(map_path):
        # if it exists, we load a temp dictionary containing the found matches
        with open(map_path) as json_file:
            temp_dict = json.load(json_file)
    # create logger for logging everything to file in case the long executions are interrupted
    # Configure the logger
    logging.basicConfig(level=logging.INFO)  # Set the desired log level
    # Create a FileHandler to write log messages to a file
    file_handler = logging.FileHandler('output.log')
    # Add the file handler to the logger
    logging.getLogger().addHandler(file_handler)

    def entity_link(asin):
        """
        It performs an HTTP request on the Wikidata API to get the Wikidata ID of the given item (referred by ASIN).

        :param asin: asin of the Amazon item for which the ID is requested
        """
        try:
            # check if the match has been already created
            # check if it is a not in dump -> in this case, we retry the search again as we could update the dump file
            # while manually checking if we missed some entities due to an incorrect query to the Wikidata Query Service
            if asin not in temp_dict or (temp_dict[asin] == retry_reason and retry):
                # check for a match if it has not been already created
                if asin in m_data:
                    # remove punctuation
                    amazon_title = re.sub("[\(\[].*?[\)\]]", "", m_data[asin])
                    amazon_title = amazon_title.replace(":", " ").replace("-", "").replace("/", "")
                    # Define parameters for the Wikidata API search
                    params = {
                        "action": "query",
                        "format": "json",
                        "list": "search",
                        "srsearch": amazon_title
                    }

                    response = requests.get(wikidata_api_url, params=params)
                    response.raise_for_status()

                    data = response.json()
                    if "search" in data["query"] and data["query"]["search"]:
                        if use_dump:
                            for item in data["query"]["search"]:
                                if item["title"] in wikidata_dump["wids"]:
                                    print("%s - %s - %s" % (asin, m_data[asin], item["title"]))
                                    logging.info("%s - %s" % (asin, item["title"]))
                                    return asin, item["title"]
                            # second loop on the alternate dump (only when processing CDs and Vinyls dataset)
                            if "CDs" in amazon_ratings:
                                # only for the CDs and Vinyls dataset, it could happen that animated movies are included
                                # in the dataset as (erroneously) treated as CDs
                                # if no entries are found in the music dump (as they should be DVDs and not CDs), we
                                # look for them in the movie dump, which contains DVDs and animated movies
                                for item in data["query"]["search"]:
                                    if item["title"] in additional_dump["wids"]:
                                        print("[alternate dump] %s - %s - %s" % (asin, m_data[asin], item["title"]))
                                        logging.info("%s - %s" % (asin, item["title"]))
                                        return asin, item["title"]
                            print("%s not found in dump" % (asin,))
                            logging.info("%s - not-in-dump" % (asin, ))
                            return asin, "not-in-dump"  # all found items are not of the correct category
                        else:
                            print("%s - %s - %s" % (asin, m_data[asin], data["query"]["search"][0]["title"]))
                            logging.info("%s - %s" % (asin, data["query"]["search"][0]["title"]))
                            return asin, data["query"]["search"][0]["title"]
                    else:
                        print("%s not found by query" % (asin,))
                        logging.info("%s - not-found-query" % (asin, ))
                        return asin, "not-found-query"  # there are no results for the query
                else:
                    print("%s does not have a title" % (asin,))
                    logging.info("%s - not-title" % (asin, ))
                    return asin, "not-title"  # the item has not a corresponding title in the metadata file
            else:
                # if the match has been already created, simply load the match
                # print("%s already matched in the mapping file" % (asin, ))
                return asin, temp_dict[asin]
        except Exception as e:
            print("%s produced the exception %s" % (asin, e))
            logging.info("%s - exception" % (asin, ))
            return asin, "exception"  # the item is not in the metadata file provided by amazon

    # here the parallel computing
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        matches = list(executor.map(entity_link, item_ids))
    # close the file handler
    file_handler.close()
    match_dict = {k: v for k, v in matches}
    with open('./data/processed/mapping-%s.json' % (amazon_ratings.split("/")[-1].split(".")[0],), 'w',
              encoding='utf-8') as f:
        json.dump(match_dict, f, ensure_ascii=False, indent=4)


def get_wid_per_cat(category):
    """
    This function executes a SPARQL query on the Wikidata Query Service to get the Wikidata ID of all the entities that
    are instance of subclasses of the given category.

    It creates a JSON file containing the list of Wikidata ID retrieved with the query.

    :param category: string representing a category name among movies, music, or books
    """
    # set the URL to the Wikidata Query Service
    endpoint_url = "https://query.wikidata.org/sparql"

    # create the correct query based on the given category
    if category == "movies":
        query = """SELECT DISTINCT ?item
        WHERE {
          {
            ?item wdt:P31/wdt:P279* wd:Q11424.
          }
          UNION
          {
            ?item wdt:P31/wdt:P279* wd:Q506240.
          }
        UNION
          {
            ?item wdt:P31/wdt:P279* wd:Q21191270.
          }
        UNION
          {
            ?item wdt:P31/wdt:P279* wd:Q24856.
          }
        UNION
          {
            ?item wdt:P31/wdt:P279* wd:Q5398426.
          }
        }"""
    elif category == "music":
        query = """SELECT DISTINCT ?item
                    WHERE {
                        ?item wdt:P31/wdt:P279* wd:Q16887380.
                    }"""
    else:
        query = """SELECT DISTINCT ?item
                    WHERE {
                        {
                        ?item wdt:P31 wd:Q47461344.
                        }
                        UNION
                        {
                        ?item wdt:P31/wdt:P279* wd:Q7725634.
                        }
                    }"""

    # execute the query
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    dump_dict = {"wids": []}

    for result in results["results"]["bindings"]:
        dump_dict["wids"].append(result["item"]["value"].split("/")[-1])

    # save Wikidata IDs to file
    with open('./data/processed/wikidata-%s.json' % (category,), 'w',
              encoding='utf-8') as f:
        json.dump(dump_dict, f, ensure_ascii=False, indent=4)


def get_no_found(metadata):
    """
    This function takes the final metadata file as input (JSON file containing an ASIN to title mapping) and produce a
    list of ASINs for which the title has not been found on Amazon neither Wayback machine.

    It creates a CSV file containing the list of ASINs without an associated title.

    :param metadata: metadata file on which we need to find the ASINs without an associated title. It could be the path
    to the metadata file or a dictionary containing the metadata
    """
    if not isinstance(metadata, dict):
        with open(metadata) as json_file:
            data = json.load(json_file)
    else:
        data = metadata
    # get the list of ASINs without a title
    no_titles = [k for k, v in data.items() if v == "404-error"]
    df = pd.DataFrame(no_titles, columns=["ASINs"])
    df.to_csv("./data/processed/item-no-titles.csv", index=False)


def count_lines(file_path):
    """
    This function efficiently counts the number of lines in a given file.

    :param file_path: path to the file
    :return: number of lines of the given file
    """
    try:
        result = subprocess.run(['wc', '-l', file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                check=True)
        output = result.stdout.decode().strip()
        line_count = int(output.split()[0])
        return line_count
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return -1


def create_wikidata_labels_sqlite(raw_labels):
    """
    This function creates a SQLITE file containing the labels of wikidata entities.

    :param raw_labels: path to raw wikidata labels file
    """
    conn = sqlite3.connect("./data/wikidata/labels.db")
    c = conn.cursor()

    c.execute('''CREATE TABLE labels (wikidata_id TEXT PRIMARY KEY, label TEXT)''')
    with open(raw_labels, 'r') as f:
        for i, line in enumerate(tqdm(f, total=count_lines(raw_labels))):
            if i != 0:
                split_line = line.split("\t")
                c.execute("INSERT INTO labels VALUES (?, ?)", (split_line[1], split_line[3]))

    conn.commit()
    conn.close()


def convert_ids_to_labels(wiki_paths_file):
    """
    This function takes a tsv file containing paths between wikidata entities and converts the IDs into actual wikidata
    labels. Note each row represents a different path.

    It creates a new tsv file containing labels instead of IDs.

    :param wiki_paths_file: path to the tsv file containing wikidata paths
    """
    df = pd.read_csv(wiki_paths_file, sep="\t")
    conn = sqlite3.connect("./data/wikidata/labels.db")
    c = conn.cursor()

    def get_label(x):
        suffix = ""
        if x.startswith("P") and x.endswith("_"):
            x = x[:-1]
            suffix = " (inverse)"
        try:
            label = c.execute("SELECT label FROM labels WHERE wikidata_id=?", (x,)).fetchone()[0]
        except TypeError:
            return x
        # remove language specification
        if "@" in label:
            label = label.split("@")[0]
        # remove ' in front and tail
        if label.startswith("'") and label.endswith("'"):
            label = label.strip("'")
        label = label + suffix
        return label

    res = df.map(lambda x: get_label(x) if isinstance(x, str) or not math.isnan(x) else "")
    res.to_csv(wiki_paths_file[:-4] + "_labelled.tsv", index=False, sep="\t")
    res = res.values
    out = {"standard": [], "inverse": []}
    for i, path in enumerate(res):
        out_str = ""
        # out_str += "path %d: " % (i + 1, )
        for item in path:
            if item != "":
                out_str = out_str + item + " ---> "
            else:
                out_str = out_str.rstrip(" ---> ")
                out_str += "\n"
                break
        out_str = out_str.rstrip(" ---> ")
        out_str += "\n"
        out_str = "(%d-hops) %s" % (int(out_str.count("--->") / 2), out_str)
        if "inverse" in out_str:
            out["inverse"].append(out_str)
        else:
            out["standard"].append(out_str)

    # create txt file of the paths
    with open(wiki_paths_file[:-4] + "_labelled.txt", 'w', encoding='utf-8') as f:
        for relation, paths in out.items():
            if paths:
                f.write(relation + "\n\n")
                f.writelines(paths)
                f.write("\n\n")

    conn.close()
