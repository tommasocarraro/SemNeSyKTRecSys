import json
import os
import random
import time
from multiprocessing import Manager

from bs4 import BeautifulSoup
from joblib import Parallel, delayed
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36"
}


def scrape_title_google_search(
    asins: list[str],
    n_cores: int = 1,
    batch_size: int = 100,
    save_tmp: bool = True,
) -> dict[str, str]:
    """
    This function takes as input a list of Amazon ASINs and performs http requests to the Google Search
    to get the title of the ASIN. It simply searches on the Google Search text area and iterates over the results to get
    the title of the searched ASIN. This is used for the ASINs that during the second scraping job (Wayback machine)
    obtained a 404 error on the Wayback machine. This is an attempt of still retrieve the title despite the official
    page of the product not existing anymore even on the Wayback Machine.

    If the script is able to find a title given the ASIN, the title is saved inside a dictionary. If the search does not
    produce any results, the script keeps track of it.

    :param asins: list of ASINs for which the title has to be retrieved
    :param n_cores: number of cores to use for multiprocessing
    :param batch_size: number of ASINs to be processed in each batch
    :param save_tmp: whether temporary retrieved title JSON files have to be saved once the batch is finished
    by Google for this specific search]
    :return: new dictionary containing key-value pairs with ASIN-title
    """
    # define dictionary suitable for parallel storing of information
    manager = Manager()
    title_dict = manager.dict()
    # Set up the Chrome options for a headless browser
    options = Options()
    options.add_argument(
        "--disable-gpu"
    )  # Disable GPU to avoid issues in headless mode
    options.add_argument(
        "--window-size=1920x1080"
    )  # Set a window size to avoid responsive design
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")

    def batch_request(
        batch_idx: int, asins: list[str], title_dict: dict[str, str]
    ) -> None:
        """
        This function performs a batch of HTTP requests to the Google Search.

        :param batch_idx: index of the current batch
        :param asins: list of ASINs of the products for which the title has to be retrieved in this batch
        :param title_dict: dictionary for storing of the retrieved data
        """
        # check if this batch has been already processed in another execution
        # if the path does not exist, we process the batch
        tmp_path = "./data/processed/tmp/metadata-batch-%s" % (batch_idx,)
        if not os.path.exists(tmp_path):
            # define dictionary for saving batch data
            batch_dict = {}
            # Set up the Chrome driver for the current batch
            chrome_service = ChromeService(executable_path="./chromedriver")
            driver = webdriver.Chrome(service=chrome_service, options=options)
            # start the scraping loop
            for counter, asin in enumerate(asins):
                print_str = ""
                # search for ASIN in Google Search
                url = "http://www.google.com/search?as_q=" + asin
                driver.get(url)
                # wait some time to avoid detection
                time.sleep(random.uniform(3, 10))
                # get page source
                soup = BeautifulSoup(driver.page_source, "html.parser")
                # check if Google changed my search
                a = soup.find("a", class_="spell_orig")
                if a is not None:
                    # get the link and go to the correct search
                    url = a.get("href")
                    driver.get("http://www.google.com" + url)
                    # get new page source
                    soup = BeautifulSoup(driver.page_source, "html.parser")
                # iterate over the links of the page
                divs = soup.find_all("div", class_="yuRUbf")
                item_title = None
                for div in divs:
                    if (
                        asin in div.a["href"]
                        and "review" not in div.a["href"]
                        and "amazon" in div.a["href"]
                    ):
                        if "..." not in div.h3.text:
                            title = div.h3.text
                            title = (
                                title.replace(" - Amazon.com", "")
                                .replace("Amazon.com: ", "")
                                .replace("Customer reviews: ", "")
                            )
                            item_title = title
                            break

                if item_title is None:
                    item_title = "404-error"
                    batch_dict[asin] = "404-error"
                else:
                    batch_dict[asin] = {
                        "title": item_title,
                        "person": None,
                        "year": None,
                    }
                print("%s - %s" % (asin, item_title))
            # Close the browser window
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
    Parallel(n_jobs=n_cores)(
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
