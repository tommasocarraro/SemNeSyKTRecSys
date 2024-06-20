import json
import re
from bs4 import BeautifulSoup
import time


def metadata_stats(metadata, errors, save_asins=False):
    """
    This function produces some statistics for the provided metadata file. The statistics include the number of items
    with a missing title due to a 404 error, a bot detection, DOM error, or an unknown error due to an exception in
    the scraping procedure. Statistics also include the number of items for which person (director, author, or artist)
    and release date are missing. This utility function is useful to have an idea of how much information is still
    missing after scraping.

    The parameter `errors` allows to define which statistics to include in the output, chosen from (404-error,
    captcha-or-DOM, exception-error).
    If save_asins is set to True, for each of these statistics, the output will contain the set of ASINs corresponding
    to each error.
    The output will also contain the percentage of items included in each statistics given the total
    number of items in the provided file.
    The output will be saved to a JSON file in the same location of the given metadata file with "_stats" added at
    the name. This, if save_asins is set to True, otherwise it just prints on the standard output.

    :param metadata: path to the metadata file for which the statistics have to be generated
    :param errors: list of strings containing the name of the errors that have to be included in the statistics. The
    script will search for these specific names in the values of the metadata, for each of the ASINs
    :param save_asins: whether to save the set of the ASINs for each statistic or just produce the statistic
    """
    link_prefix = "https://www.amazon.com/dp/"
    errors = {e: {"counter": 0, "asins": []} for e in errors}
    with open(metadata) as json_file:
        m_data = json.load(json_file)
    for asin, data in m_data.items():
        if type(data) is not dict:
            if data in errors:
                errors[data]["counter"] += 1
                if save_asins:
                    errors[data]["asins"].append(link_prefix + asin)
        else:
            if "matched" in errors:
                errors["matched"]["counter"] += 1
            else:
                errors["matched"] = {"counter": 1}

            if data["person"] is None:
                if "person" not in errors:
                    errors["person"] = {"counter": 1}
                    if save_asins:
                        errors["person"]["asins"] = [link_prefix + asin]
                else:
                    errors["person"]["counter"] += 1
                    if save_asins:
                        errors["person"]["asins"].append(link_prefix + asin)

            if data["year"] is None:
                if "year" not in errors:
                    errors["year"] = {"counter": 1}
                    if save_asins:
                        errors["year"]["asins"] = [link_prefix + asin]
                else:
                    errors["year"]["counter"] += 1
                    if save_asins:
                        errors["year"]["asins"].append(link_prefix + asin)

            if data["year"] is None and data["person"] is None:
                if "person+year" not in errors:
                    errors["person+year"] = {"counter": 1}
                    if save_asins:
                        errors["person+year"]["asins"] = [link_prefix + asin]
                else:
                    errors["person+year"]["counter"] += 1
                    if save_asins:
                        errors["person+year"]["asins"].append(link_prefix + asin)

    # compute percentages
    # total = sum([errors[e]["counter"] for e in errors])
    # for e in errors:
    #     errors[e]["percentage"] = errors[e]["counter"] / total * 100
    if save_asins:
        with open(metadata[:-5] + "_stats.json", 'w', encoding='utf-8') as f:
            json.dump(errors, f, ensure_ascii=False, indent=4)
    else:
        print(errors)


def scrape_metadata(driver, batch_dict, asin, url, bot_counter):
    """
    This function takes as input a selenium driver with a loaded Amazon product page and search for metadata inside the
    page.

    :param driver: selenium driver with the loaded page
    :param batch_dict: dictionary that has to be updated with the new metadata
    :param asin: asin of the product for which the metadata has to be scraped
    :param url: url of the amazon product page (for debugging purposes)
    :param bot_counter: counter of number of detections
    :return: the bot counter and a string to be printed in case of exception
    """
    # get the page
    page = driver.page_source
    # Parse the HTML content of the page
    soup = BeautifulSoup(page, "html.parser")
    # check if there is a bot detection from Amazon
    # if soup.find("i", {"class": "a-icon-alert"}):
    #     # give time to manually solve captcha
    #     print("captcha detected, fill it!")
    #     time.sleep(10)
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
    return bot_counter, print_str
