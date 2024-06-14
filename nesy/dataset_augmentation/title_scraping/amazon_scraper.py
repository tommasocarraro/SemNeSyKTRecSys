import json
import os
import random
import re
import time
from multiprocessing import Manager

from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from joblib import Parallel, delayed
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService

def scrape_title_amazon(
    asins: list[str], n_cores: int, batch_size: int, save_tmp: bool = True, batch_i_start: int = 0, batch_i_end: int =0
) -> dict[str, str]:
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
    :param batch_i_start: the batch index from which the scraping has to be started. This is useful to start the scraping
    on multiple machines so that they can perform part of the scraping (kind of distributed computation)
    :param batch_i_end: the index of the last batch of the scraping job. The job will execute all the batches from the
    start index to this index
    :return: new dictionary containing key-value pairs with scraped ASIN-title
    """
    # get number of batches for printing information
    n_batches = int(len(asins) / batch_size)
    # define dictionary suitable for parallel storing of information
    manager = Manager()
    title_dict = manager.dict()
    # Set up the Chrome options for a headless browser
    chrome_options = Options()
    chrome_options.add_argument(
        "--disable-gpu"
    )  # Disable GPU to avoid issues in headless mode
    chrome_options.add_argument(
        "--window-size=1920x1080"
    )  # Set a window size to avoid responsive design
    ua = UserAgent()
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")

    def batch_request(
        batch_idx: int, asins: list[str], title_dict: dict[str, str]
    ) -> None:
        """
        This function performs a batch of HTTP requests using Selenium.

        :param batch_idx: index of the current batch
        :param asins: list of ASINs of the products for which the title has to be retrieved in this batch
        :param title_dict: dictionary for parallel storing of the retrieved data
        """
        if batch_i_start <= batch_idx <= batch_i_end:
            # check if this batch has been already processed in another execution
            # if the path does not exist, we process the batch
            tmp_path = "./data/processed/tmp/metadata-batch-%s" % (batch_idx,)
            if not os.path.exists(tmp_path) or not save_tmp:
                # set the user agent
                chrome_options.add_argument(f"user-agent={ua.random}")  # random user agent
                # define dictionary for saving batch data
                batch_dict = {}
                # create the URLs for scraping
                urls = [f"https://www.amazon.com/dp/{asin}" for asin in asins]
                # Set up the Chrome driver for the current batch
                chrome_service = ChromeService(executable_path="./chromedriver")
                driver = webdriver.Chrome(service=chrome_service, options=chrome_options)
                bot_counter = 0
                # start the scraping loop
                for counter, url in enumerate(urls):
                    asin = url.split("/")[-1]
                    try:
                        # wait some time to avoid detection
                        time.sleep(random.uniform(1, 3))
                        # Load the Amazon product page
                        driver.get(url)
                        # get the page
                        page = driver.page_source
                        # Parse the HTML content of the page
                        soup = BeautifulSoup(page, "html.parser")
                        # Find the product title
                        title_element = soup.find("span", {"id": "productTitle"})
                        if title_element:
                            bot_counter = 0
                            # the title has been found and we save it in the dictionary
                            print_str = title_element.text.strip()
                            batch_dict[asin] = {}
                            batch_dict[asin]["title"] = title_element.text.strip()
                            # get person information
                            person = soup.find("span", {"class": "author"})
                            # this is used in a second step
                            person_year_div = soup.find(
                                "div", {"id": "detailBullets_feature_div"}
                            )
                            if person:
                                batch_dict[asin]["person"] = person.a.text
                            else:
                                found_person = False
                                if person_year_div:
                                    spans = person_year_div.find_all("span")
                                    for i, span in enumerate(spans):
                                        if "Director" in span.text:
                                            batch_dict[asin]["person"] = spans[i + 2].text
                                            found_person = True
                                            break
                                        if "Actors" in span.text:
                                            batch_dict[asin]["person"] = spans[i + 2].text
                                            found_person = True
                                            break
                                    if not found_person:
                                        # look for contributor
                                        person = soup.find(
                                            "tr", {"class": "po-contributor"}
                                        )
                                        if person:
                                            person = person.find(
                                                "span", {"class": "po-break-word"}
                                            )
                                            if person:
                                                batch_dict[asin]["person"] = person.text
                                            else:
                                                batch_dict[asin]["person"] = None
                                        else:
                                            batch_dict[asin]["person"] = None
                                else:
                                    batch_dict[asin]["person"] = None
                            # get year information
                            found_year = False
                            if person_year_div:
                                spans = person_year_div.find_all("span")
                                year_pattern = re.compile(r"\b\d{4}\b")
                                for span in spans:
                                    year_match = year_pattern.search(span.text)
                                    if year_match:
                                        batch_dict[asin]["year"] = year_match.group()
                                        found_year = True
                                        break
                                if not found_year:
                                    batch_dict[asin]["year"] = None
                            else:
                                batch_dict[asin]["year"] = None
                        else:
                            # the title has not been found
                            # check if it is due to a 404 error
                            error = soup.find(
                                "img",
                                {
                                    "alt": "Sorry! We couldn't find that page. "
                                    "Try searching or go to Amazon's home page."
                                },
                            )
                            if error:
                                # if it is due to 404 error, keeps track of it
                                # items not found will be processed in another scraping loop that uses Wayback Machine
                                print_str = "404 error - url %s" % (url,)
                                batch_dict[asin] = "404-error"
                            else:
                                bot_counter += 1
                                if bot_counter == 20:
                                    raise Exception("Bot detection")
                                # if it is not a 404 error, the bot has been detected
                                print_str = "Bot detected - url %s" % (url,)
                                # it could be because of a captcha from Amazon or also because there is not productTitle
                                # but another ID
                                # items bot-detected will be processed in another scraping loop that tries again
                                # if the problem is related to the DOM, the web page has to be investigated
                                batch_dict[asin] = "captcha-or-DOM"
                    except Exception as e:
                        if bot_counter == 20:
                            raise Exception("Bot detection")
                        print(e)
                        print_str = "unknown error - url %s" % (url,)
                        # if an exception is thrown by the system, I am interested in knowing which ASIN caused that, so
                        # I keep track of the exception in the dictionary
                        batch_dict[asin] = "exception-error"
                    print(
                        "batch %d / %d - asin %d / %d - %s"
                        % (batch_idx, n_batches, counter, len(asins), print_str)
                    )
                # after each batch, the resources allocated by Selenium have to be realised
                driver.quit()
                if save_tmp:
                    # save a json file dedicated to this specific batch
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        json.dump(batch_dict, f, ensure_ascii=False, indent=4)
            else:
                # load the file and update the parallel dict
                with open(tmp_path) as json_file:
                    batch_dict = json.load(json_file)
            # update parallel dict
            title_dict.update(batch_dict)

    # create folder for saving temporary data
    if save_tmp:
        if not os.path.exists("./data/processed/tmp"):
            os.makedirs("./data/processed/tmp")

    # parallel scraping -> asins are subdivided into batches and the batches are run in parallel
    Parallel(n_jobs=n_cores, prefer="threads")(
        delayed(batch_request)(batch_idx, a, title_dict)
        for batch_idx, a in enumerate(
            [
                asins[
                    i : (i + batch_size if i + batch_size < len(asins) else len(asins))
                ]
                for i in range(0, len(asins), batch_size)
            ]
        )
    )
    return title_dict.copy()
