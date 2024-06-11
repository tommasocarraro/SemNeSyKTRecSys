import json
import os
import time

import wayback
from bs4 import BeautifulSoup


def scrape_title_wayback(
    asins: list[str], batch_size: int = 20, save_tmp: bool = True, delay: int = 60
) -> dict[str, str]:
    """
    This function takes as input a list of Amazon ASINs and performs http requests with an unofficial Wayback API to get
    the title of the ASIN from the Wayback Machine website. This is used for the ASINs that during the first scraping
    job on the Amazon website obtained a 404 error. This is an attempt of still retrieve the title despite the official
    page of the product not existing anymore. Note that parallel execution is not possible with Wayback Machine as they
    are blocking for one minute every 20 HTTP requests.

    If the script is able to find a title given the ASIN, the title is saved inside a dictionary. If a captcha or 404
    error is detected even in the Wayback Machine, the script keeps track of it. Finally, if there is a DOM problem,
    namely the title cannot be found due to an ID not found in the DOM of the web page, the script keeps track of it
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

    def batch_request(
        batch_idx: int, asins: list[str], title_dict: dict[str, str]
    ) -> None:
        """
        This function performs a batch of HTTP requests using an unofficial Wayback API.

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
            # define the Amazon URLs that have to be searched in the Wayback Machine
            urls = [f"https://www.amazon.com/dp/{asin}" for asin in asins]
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
                        soup = BeautifulSoup(page, "html.parser")  # html.parser
                        # check if the page is a captcha page and discard it from the search
                        if soup.find("h4") is not None and soup.find(
                            "h4"
                        ).get_text() == ("Enter the characters you " "see below"):
                            batch_dict[asin] = "captcha"
                            print_str = "captcha problem - ASIN %s" % (url,)
                            # move to next snapshot in the for loop
                            continue
                        # check if it is a saved page with 404 error
                        if (
                            soup.find("b", {"class": "h1"}) is not None
                            and "Looking for something?"
                            in soup.find("b", {"class": "h1"}).get_text()
                        ) or soup.find(
                            "img",
                            {
                                "alt": "Sorry! We couldn't find that page. "
                                "Try searching or go to Amazon's home page."
                            },
                        ) is not None:
                            batch_dict[asin] = "404-error"
                            print_str = "404 error - ASIN %s" % (url,)
                            # move to next snapshot in the for loop
                            continue
                        # if it is not a captcha page or 404 page, find this specific IDs (in order) while
                        # parsing the web page
                        id_alternatives = [
                            "btAsinTitle",
                            "productTitle",
                            "ebooksProductTitle",
                        ]
                        title_element = soup.find("span", {"id": id_alternatives[0]})
                        i = 1
                        while title_element is None and i < len(id_alternatives):
                            title_element = soup.find(
                                "span", {"id": id_alternatives[i]}
                            )
                            i += 1
                        if title_element is not None:
                            batch_dict[asin] = title_element.text.strip()
                            print_str = title_element.text.strip()
                            # one has been found, no need to continue the search
                            break
                        else:
                            # if none of the IDs has been found, the page is very old and we need to search for
                            # a "b" with class "sans"
                            title_element = soup.find("b", {"class": "sans"})
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
                print(
                    "batch %d / %d - asin %d / %d - %s"
                    % (batch_idx, n_batches, counter, len(asins), print_str)
                )
            if save_tmp:
                # save a json file dedicated to this specific batch
                with open(tmp_path, "w", encoding="utf-8") as f:
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
    for batch_idx, batch_asins in enumerate(
        [
            asins[i : (i + batch_size if i + batch_size < len(asins) else len(asins))]
            for i in range(0, len(asins), batch_size)
        ]
    ):
        batch_request(batch_idx, batch_asins, title_dict)
    return title_dict.copy()
