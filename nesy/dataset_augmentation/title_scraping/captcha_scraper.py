from joblib import Parallel, delayed
import json
from multiprocessing import Manager
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time


def scrape_title_captcha(asins: list[str],
                         n_cores: int = 1,
                         batch_size: int = 100,
                         save_tmp: bool = True,
                         delay: int = 180,
                         use_solver: bool = True) -> dict[str, str]:
    """
    This function takes as input a list of Amazon ASINs and performs http requests to the Rocket Source knowledge base
    to get the title of the ASIN. This is used for the ASINs that during the second scraping job (Wayback machine)
    obtained a 404 error on the Wayback machine. This is an attempt of still retrieve the title despite the official
    page of the product not existing anymore even on the Wayback Machine. Note the site requires to solve captchas, so
    a captcha solver has to be used in this function. NoCaptchaAI is used here.

    If the script is able to find a title given the ASIN, the title is saved inside a dictionary. If the ASIN is not
    included in the database, the script keeps track of it.

    :param asins: list of ASINs for which the title has to be retrieved
    :param n_cores: number of cores to use for multiprocessing
    :param batch_size: number of ASINs to be processed in each batch
    :param save_tmp: whether temporary retrieved title JSON files have to be saved once the batch is finished
    :param delay: number of seconds to wait for solving captchas. It is possible the tool requires a lot of time for
    solving a CAPTCHA, depending on how much challenging it is.
    :param use_solver: whether to use a captcha solver or not. If not, the user will have to solve captchas autonomously
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
    if use_solver:
        options.add_extension("./nocaptcha/noCaptchaAi-chrome-v1.3.crx")
    # url for configuring the extension for captchas
    url_api = "https://newconfig.nocaptchaai.com/?APIKEY=bmxitalia-5f6ff273-aa2e-2ffb-fff7-c0388051fe71&PLANTYPE=pro&customEndpoint=&hCaptchaEnabled=true&reCaptchaEnabled=true&dataDomeEnabled=true&ocrEnabled=true&ocrToastEnabled=true&extensionEnabled=true&logsEnabled=false&fastAnimationMode=true&debugMode=false&hCaptchaAutoOpen=true&hCaptchaAutoSolve=true&hCaptchaAlwaysSolve=true&englishLanguage=true&hCaptchaGridSolveTime=7&hCaptchaMultiSolveTime=5&hCaptchaBoundingBoxSolveTime=5&reCaptchaAutoOpen=true&reCaptchaAutoSolve=true&reCaptchaAlwaysSolve=true&reCaptchaClickDelay=400&reCaptchaSubmitDelay=1&reCaptchaSolveType=null"
    # url of the webpage of Rocket Source knowledge base
    url = "https://www.rocketsource.io/asin-to-ean"

    def batch_request(batch_idx: int, asins: list[str], title_dict: dict[str, str]) -> None:
        """
        This function performs a batch of HTTP requests to the Rocket Source knowledge base.

        :param batch_idx: index of the current batch
        :param asins: list of ASINs of the products for which the title has to be retrieved in this batch
        :param title_dict: dictionary for storing of the retrieved data
        """
        # check if this batch has been already processed in another execution
        # if the path does not exist, we process the batch
        tmp_path = "./data/processed/tmp/metadata-batch-%s" % (batch_idx,)
        if not os.path.exists(tmp_path) or not save_tmp:
            # define dictionary for saving batch data
            batch_dict = {}
            # Set up the Chrome driver for the current batch
            driver = webdriver.Chrome(executable_path="./chromedriver", options=options)
            if use_solver:
                # config the extension
                driver.get(url_api)
            # start the scraping loop
            for counter, asin in enumerate(asins):
                # open the webpage - this needs to be open everytime because it has to load everything again
                # this is a requirement because the Javascript loads the title once the Convert button is pressed
                # if the page is not reloaded, the title remains there and the script takes the same title
                # multiple times
                driver.get(url)
                # Find the input element where the ASIN has to be inserted
                input_element = driver.find_element(By.TAG_NAME, "input")
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
    Parallel(n_jobs=n_cores)(delayed(batch_request)(batch_idx, a, title_dict)
                             for batch_idx, a in
                             enumerate([asins[i:(i + batch_size if i + batch_size < len(asins) else len(asins))]
                                        for i in range(0, len(asins), batch_size)]))
    return title_dict.copy()
