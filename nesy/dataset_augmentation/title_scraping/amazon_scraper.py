from joblib import Parallel, delayed
import json
from multiprocessing import Manager
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import os


def scrape_title_amazon(asins: list[str], n_cores: int, batch_size: int, save_tmp: bool = True) -> dict[str, str]:
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

    def batch_request(batch_idx: int, asins: list[str], title_dict: dict[str, str]) -> None:
        """
        This function performs a batch of HTTP requests using Selenium.

        :param batch_idx: index of the current batch
        :param asins: list of ASINs of the products for which the title has to be retrieved in this batch
        :param title_dict: dictionary for parallel storing of the retrieved data
        """
        # check if this batch has been already processed in another execution
        # if the path does not exist, we process the batch
        tmp_path = "./data/processed/metadata-batch-%s" % (batch_idx,)
        if not os.path.exists(tmp_path) or not save_tmp:
            # define dictionary for saving batch data
            batch_dict = {}
            # create the URLs for scraping
            urls = [f'https://www.amazon.com/dp/{asin}' for asin in asins]
            # Set up the Chrome driver for the current batch
            driver = webdriver.Chrome(executable_path="./chromedriver", options=chrome_options)
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
                            print_str = "404 error - url %s" % (url,)
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
                    print_str = "unknown error - url %s" % (url,)
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
