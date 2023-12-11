import ast
import os
from joblib import Parallel, delayed
import json
from multiprocessing import Manager
import csv
import pandas as pd
import requests
from tqdm import tqdm
import re
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options


def create_asin_metadata_json(metadata):
    """
    This function takes as input the JSON file containing Amazon product metadata and produces a filtered version of
    the same file, where for each ASIN we store information about just the title.

    The new file is saved into the /processed folder.

    :param metadata: path of JSON file containing Amazon product metadata
    """
    manager = Manager()
    md_dict = manager.dict()

    def process_line(line, md_dict):
        """
        Function used to process one line of the big Amazon metadata JSON file. It processes the line and saves only
        the relevant information into the given dictionary. The dictionary is indexed by the ASIN of the products.

        When no information is available on the metadata file, it keeps track of it for then using scraping to scrape
        relevant information from the web.

        :param line: line of JSON file containing metadata
        :param md_dict: dictionary containing only relevant metadata
        """
        row_dict = ast.literal_eval(line)
        try:
            md_dict[row_dict["asin"]] = row_dict["title"]
        except Exception as e:
            print(e)
            md_dict[row_dict["asin"]] = "no-title"

    with open(metadata) as f:
        Parallel(n_jobs=os.cpu_count())(delayed(process_line)(line, md_dict) for line in f)

    with open('./data/processed/%s' % (metadata.split("/")[-1]), 'w', encoding='utf-8') as f:
        json.dump(md_dict.copy(), f, ensure_ascii=False, indent=4)


def filter_metadata(metadata, rating_files):
    """
    This function filter the given metadata file using the item IDs found in the given rating files.
    The filtering process just creates a new metadata file where only the items included in the provided rating files
    are kept.
    It creates a new metadata file in the same location of metadata, called filtered-metadata.

    :param metadata: path to metadata file
    :param rating_files: list of rating file names. Provide only the name (e.g., reviews_Movies_and_TV_5)
    """
    desired_asin = []
    for dataset in rating_files:
        data = pd.read_csv("./data/processed/%s.csv" % (dataset,))
        desired_asin.extend(data["itemId"].unique())

    with open(metadata) as json_file:
        m_data = json.load(json_file)

    new_file_dict = {k: m_data[k] for k in desired_asin}

    with open('./data/processed/filtered-%s' % (metadata.split("/")[-1]), 'w', encoding='utf-8') as f:
        json.dump(new_file_dict, f, ensure_ascii=False, indent=4)


def scrape_title(asins, n_cores, batch_size, save_tmp=True):
    """
    This function takes as input a list of Amazon ASINs and performs http requests with Selenium to get the title of
    the ASIN from the Amazon website.

    :param asins: list of ASINs for which the title has to be retrieved
    :param n_cores: number of cores to be used for scraping
    :param batch_size: number of ASINs to be processed in each batch of parallel execution
    :param save_tmp: whether temporary retrieved title JSON files have to be saved once the batch is finished
    :return: new dictionary containing key-value pairs with ASIN-title
    """
    # get number of batches for printing information
    n_batches = int(len(asins) / batch_size)
    # define dictionary suitable for parallel storing of information
    manager = Manager()
    title_dict = manager.dict()
    # Set up the Chrome options for a headless browser
    chrome_options = Options()
    # headless mode causes Amazon to detect the scraping tool
    # chrome_options.add_argument('--headless')
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
        tmp_path = "./data/processed/metadata-batch-%s" % (batch_idx, )
        if not os.path.exists(tmp_path):
            # define dictionary for saving batch data
            batch_dict = {}
            urls = [f'https://www.amazon.com/dp/{asin}' for asin in asins]
            # define counter to have idea of the progress of the process
            # Set up the Chrome driver for the current batch
            chrome_service = ChromeService(executable_path='./chromedriver')
            driver = webdriver.Chrome(service=chrome_service, options=chrome_options)
            # options for wayback machine
            # timestamp = 20150101000001
            # wayback_url = "http://archive.org/wayback/available"
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
                        print_str = title_element.text.strip()
                        batch_dict[asin] = title_element.text.strip()
                    else:
                        # if title element does not exist, it means the bot has been detected or the ASIN does not exist o
                        # n Amazon
                        # check if it is a 404 error
                        title_element = soup.find('img', {'alt': "Sorry! We couldn't find that page. "
                                                                 "Try searching or go to Amazon's home page."})
                        if title_element:
                            print_str = "404 error"
                            # params = {"url": url}
                            # response = requests.get(wayback_url, params=params)
                            # data = response.json()
                            # archived_snapshots = data.get("archived_snapshots")
                            # if archived_snapshots:
                            #     new_url = archived_snapshots.get("closest", {}).get("url")
                            #     driver.get(new_url)
                            #     # get the page
                            #     page = driver.page_source
                            #     # Parse the HTML content of the page
                            #     soup = BeautifulSoup(page, 'html.parser')
                            #     # Find the product title
                            #     title_element = soup.find('span', {'id': 'productTitle'})
                            #     if title_element:
                            #         print("Found title in old version of the page - %s" % (title_element.text.strip()))
                            #         title_dict[asin] = title_element.text.strip()
                            #     else:
                            #         print("no found title in old version of page")
                            # else:
                            #     print("404 error even in wayback")
                            #     title_dict[asin] = "404-error"
                            batch_dict[asin] = "404-error"
                        else:
                            # if it is not a 404 error, the bot has been detected
                            print_str = "Bot detected - url %s" % (url,)
                            # it could be because of a captcha from Amazon or also because it is note productTile
                            # but another ID
                            batch_dict[asin] = "captcha-or-DOM"
                except Exception as e:
                    print_str = "unknown error"
                    # if an exception is thrown by the system I am interested in knowing which ASIN caused that
                    batch_dict[asin] = "exception-error"
                print("batch %d / %d - asin %d / %d - %s" % (batch_idx, n_batches, counter, len(asins), print_str))
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

    Parallel(n_jobs=n_cores)(delayed(batch_request)(batch_idx, a, title_dict)
                             for batch_idx, a in
                             enumerate([asins[i:(i + batch_size if i + batch_size < len(asins) else len(asins))]
                                        for i in range(0, len(asins), batch_size)]))
    return title_dict.copy()


def metadata_stats(metadata):
    """
    This function produces some statistics for the provided metadata file. The statistics include the number of items
    with a missing title due to a 404 error, a bot detection or DOM error, or an unknown error due to an exception in
    the scraping procedure.

    :param metadata: path to the metadata file for which the statistics have to be generated
    """
    with open(metadata) as json_file:
        m_data = json.load(json_file)
    err, captcha, ukn, matched = 0, 0, 0, 0
    for _, v in m_data.items():
        if v == "404-error":
            err += 1
        elif v == "captcha-or-DOM":
            captcha += 1
        elif v == "exception-error":
            ukn += 1
        else:
            matched += 1

    print("Matched titles %d - 404 err %d - captcha or DOM %d - exception %d" % (matched, err, captcha, ukn))
    print("Total is %d / %d" % (matched + err + captcha + ukn, len(m_data)))


def metadata_scraping(metadata, n_cores, motivation="no-title"):
    """
    This function takes as input a metadata file with some missing titles and uses web scraping to retrieve these
    titles from the Amazon website.

    At the end, a new metadata file with complete information is generated.

    :param metadata: file containing metadata for the items
    :param n_cores: number of cores to be used for scraping
    :param motivation: motivation for which the title has to be scraped. It could be:
        - no-title (default): in the first scraping job, no-title means that the metadata is missing the title for the
        item
        - captcha-or-DOM: in the first scraping job, the title retrieval for this item failed due to incorrect DOM or
        bot detection from Amazon
        - 404-error: in the first scraping job, the title retrieval failed due to a 404 not found error, meaning the
        ASIN is not on Amazon anymore
        - exception-error: in the first scraping job, the title retrieval failed due to an exception
    """
    with open(metadata) as json_file:
        m_data = json.load(json_file)
    # take the ASINs for the products that have a missing title in the metadata file
    no_titles = [k for k, v in m_data.items() if v == motivation]
    # update the metadata with the scraped titles
    m_data = m_data | scrape_title(no_titles, n_cores, batch_size=100)
    # generate the new and complete metadata file
    with open('./data/processed/complete-%s' % (metadata.split("/")[-1]), 'w', encoding='utf-8') as f:
        json.dump(m_data, f, ensure_ascii=False, indent=4)


def create_pandas_dataset(data):
    """
    This function takes as input the JSON file containing Amazon reviews and produces a CSV file. Each record has
    user ID, item ID, rating, timestamp. In other words, it creates a rating file based on the given review file.

    The new file is saved into the /processed folder.

    :param data: path of JSON file containing Amazon reviews.
    """
    manager = Manager()
    csv_list = manager.list()

    def process_line(line, rating_list):
        """
        Function used to process one line of the Amazon review file. It processes the line and saves only
        the relevant information into the given dictionary.

        :param line: line of JSON file containing ratings
        :param rating_list: list containing ratings to be saved in the CSV file. Each rating is a dict.
        """
        if "large" not in data:
            row_dict = ast.literal_eval(line)
        else:
            row_dict = json.loads(line)
        try:
            rating_list.append({"userId": row_dict["reviewerID"], "itemId": row_dict["asin"],
                                "rating": row_dict["overall"], "timestamp": row_dict["unixReviewTime"]})
        except:
            pass

    with open(data) as f:
        Parallel(n_jobs=os.cpu_count())(delayed(process_line)(line, csv_list) for line in tqdm(f))

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
    accessed using HTTP requests. It takes as input a CSV file containing ratings on Amazon items. Depending on the
    rating file (movies, music, or books), the function uses the correct JSON dump file to check the validity of the
    found match.

    It creates a JSON file containing the mapping from Amazon ASIN to Wikidata ID.

    :param amazon_ratings: CSV file containing ratings on Amazon items
    :param use_dump: whether a JSON dump has to be used to check that the retrieved ID is of the correct category
    :param query: whether to use the action "query" of the wikidata API or not
    """
    # read data
    data = pd.read_csv(amazon_ratings)
    # get item IDs
    item_ids = data['itemId'].unique()
    # read metadata
    with open("./data/processed/metadata.json") as json_file:
        m_data = json.load(json_file)
    # get wikidata dump
    wikidata_dump = ""
    if "Movies" in amazon_ratings:
        wikidata_dump = "./data/processed/wikidata-movies.json"
    if "Books" in amazon_ratings:
        wikidata_dump = ".data/processed/wikidata-books.json"
    if "CD" in amazon_ratings:
        wikidata_dump = "./data/processed/wikidata-music.json"
    with open(wikidata_dump) as json_file:
        wikidata_dump = json.load(json_file)
    # link to API
    wikidata_api_url = "https://www.wikidata.org/w/api.php"

    def entity_link(asin):
        """
        It performs an HTTP request on the Wikidata API to get the Wikidata ID of the given item (referred by ASIN).

        :param asin: asin of the Amazon item for which the ID is requested
        """
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
                for item in data["search"]:
                    if item["id"] in wikidata_dump:
                        match_dict[asin] = item["id"]
                        break
            else:
                match_dict[asin] = data["search"][0]["id"]
        else:
            match_dict[asin] = ""

    # here the parallel computing
    manager = Manager()
    match_dict = manager.dict()
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        executor.map(entity_link, item_ids)
    match_dict = match_dict.copy()
    with open('./data/processed/mapping-%s' % (amazon_ratings.split("/")[-1],), 'w', encoding='utf-8') as f:
        json.dump(match_dict, f, ensure_ascii=False, indent=4)


def entity_linker_api_query(amazon_ratings, use_dump=True):
    """
    This function uses the Wikidata API (action=query) to get the ID of wikidata items corresponding to the Amazon
    items. The API is accessed using HTTP requests. It takes as input a CSV file containing ratings on Amazon items.
    Depending on the rating file (movies, music, or books), the function uses the correct JSON dump file to check the
    validity of the found match.

    It creates a JSON file containing the mapping from Amazon ASIN to Wikidata ID.

    :param amazon_ratings: CSV file containing ratings on Amazon items
    :param use_dump: whether a JSON dump has to be used to check that the retrieved IDs are of the correct category
    (movies, music, or books)
    """
    # read data
    data = pd.read_csv(amazon_ratings)
    # get item IDs
    item_ids = data['itemId'].unique()
    # read metadata
    with open("./data/processed/metadata.json") as json_file:
        m_data = json.load(json_file)
    # get wikidata dump
    wikidata_dump = ""
    if "Movies" in amazon_ratings:
        wikidata_dump = "./data/processed/wikidata-movies.json"
    if "Books" in amazon_ratings:
        wikidata_dump = ".data/processed/wikidata-books.json"
    if "CD" in amazon_ratings:
        wikidata_dump = "./data/processed/wikidata-music.json"
    with open(wikidata_dump) as json_file:
        wikidata_dump = json.load(json_file)
    # link to API
    wikidata_api_url = "https://www.wikidata.org/w/api.php"

    def entity_link(asin):
        """
        It performs an HTTP request on the Wikidata API to get the Wikidata ID of the given item (referred by ASIN).

        :param asin: asin of the Amazon item for which the ID is requested
        """
        try:
            if m_data[asin] != "no-title":
                amazon_title = re.sub("[\(\[].*?[\)\]]", "", m_data[asin])
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
                            if item["title"] in wikidata_dump:
                                return asin, item["title"]
                        return asin, "not-in-wikidata"  # not in wikidata because all found items are not of the correct category
                    else:
                        return asin, data["query"]["search"][0]["title"]
                else:
                    return asin, "not-in-wikidata"  # not in wikidata because there are no results for the query
            else:
                return asin, "no-title"  # the item has not a corresponding title in the metadata file
        except Exception as e:
            print(f"Error: {e}")
            return asin, "no-metadata"  # the item is not in the metadata file provided by amazon

    # here the parallel computing
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        matches = list(executor.map(entity_link, item_ids))
    match_dict = {k: v for k, v in matches}
    with open('./data/processed/mapping-%s.json' % (amazon_ratings.split("/")[-1].split(".")[0],), 'w',
              encoding='utf-8') as f:
        json.dump(match_dict, f, ensure_ascii=False, indent=4)
