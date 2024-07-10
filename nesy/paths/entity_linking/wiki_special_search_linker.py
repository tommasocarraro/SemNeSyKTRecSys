import logging
import json
import os
import re
import requests
from joblib import Parallel, delayed
import traceback


def entity_linker_api_query(amazon_metadata: str, mapping_file: str, retry_reason: str = None, n_cores: int = 1) -> None:
    """
    This function uses the Wikidata API (action=query) to get the ID of wikidata entities corresponding to the Amazon
    items in the given Amazon metadata. The API is accessed using HTTP requests.

    For each item in the metadata, Wikidata is queried with all the available information (title, person, and year).
    If no matches are found, a new query with just title and person is used. Then, a query with title and year. Finally,
    a query with just the title. If even this query fails, the match is discarded. When not all this information is
    available, the available information is used instead.

    Depending on the type of metadata in input, the algorithm also uses an additional file to check whether the matched
    item belongs to the correct category. This file simply contains a list of wikidata IDs belonging to a specific
    category. For example, if we are matching movies, this file contains the list of all the movies in wikidata. This
    is useful as the music dataset contains many musical DVDs too.

    At the end, a JSON file containing the mapping from Amazon ASIN to Wikidata ID is created and saved at mapping_file.

    It is possible to launch this function again by specifying a retying reason, for example "not-found-query".

    :param amazon_metadata: JSON file containing metadata on Amazon items
    :param mapping_file: path where to save the final mapping json file
    :param retry_reason: string indicating for which items the search has to be computed again. For example, indicate
    "exception" for the items that gave an exception during the first mapping loop. The procedure will repeat the loop
    only for these items.
    :param n_cores: number of processors to be used to execute this function
    """
    # read metadata
    with open(amazon_metadata) as json_file:
        m_data = json.load(json_file)
    # get correct category item list for checking matched items
    correct_category_items = ""
    if "movies" in amazon_metadata:
        correct_category_items = "./data/processed/legacy/wikidata-movies.json"
    if "books" in amazon_metadata:
        correct_category_items = "./data/processed/legacy/wikidata-books.json"
    if "music" in amazon_metadata:
        correct_category_items = "./data/processed/legacy/wikidata-music.json"
    with open(correct_category_items) as json_file:
        correct_category_items = json.load(json_file)
    # link to API for queries
    wikidata_api_url = "https://www.wikidata.org/w/api.php"
    # check if a mapping file for the given metadata already exists
    temp_dict = {}
    if os.path.exists(mapping_file):
        # if it exists, we load a temp dictionary containing the found matches
        with open(mapping_file) as json_file:
            temp_dict = json.load(json_file)
    # create logger for logging everything to file in case the long executions are interrupted
    # Configure the logger
    logging.basicConfig(level=logging.INFO)  # Set the desired log level
    # Create a FileHandler to write log messages to a file
    file_handler = logging.FileHandler('output.log')
    # Add the file handler to the logger
    logging.getLogger().addHandler(file_handler)

    def entity_link(item_metadata: tuple):
        """
        It performs an HTTP request on the Wikidata API to get the Wikidata ID of the given item, for which the
        metadata is given.

        :param item_metadata: metadata of the Amazon item for which the ID is requested
        """
        try:
            asin, metadata = item_metadata
            # check if the match has been already created
            if (asin not in temp_dict or
                    (retry_reason is not None and temp_dict[asin] == retry_reason)):
                # check if metadata is available
                if isinstance(metadata, dict):
                    year = None
                    if metadata["year"] is None:
                        # try to get year from title, if available
                        year_groups = re.search(r"\((19\d{2}|20\d{2})\)|\[(19\d{2}|20\d{2})\]",
                                                metadata["title"])
                        if year_groups is not None:
                            year = year_groups.group(1) or year_groups.group(2)
                            year = int(year)
                    else:
                        year = metadata["year"]
                    # remove tags between square and round braces
                    title = re.sub(r"[\[\(].*?[\]\)]", "", metadata["title"])
                    title = title.rstrip()
                    # remove tags without braces
                    if title.endswith("DVD") or title.endswith("VHS"):
                        title = title[:-3].rstrip()
                    # remove explicit lyrics warning
                    if title.endswith("explicit_lyrics"):
                        title = title[: -len("explicit_lyrics")].rstrip()
                    title_before_regex = title
                    # remove common patterns
                    all_pattern = re.compile(
                        r"^.*?(?=\s*(?:\bthe\b\s)?[.,:-]?\s?(?:\bvolume\b|\bseason\b|\bvol\b\.?|\bcomplete\b|\bprograms\b|\bset\b|("
                        r"?:\s*\b\w*[^\s\w]\w*\b|\b\w+\b\s*){0,2}\bedition\b|\bcollection\b\s*$|\bcollector("
                        r"?:\'s)?\b\s\b\w*\b|\bwidescreen\b)|\s*$)",
                        re.IGNORECASE,
                    )
                    groups = re.match(all_pattern, title)
                    if groups:
                        title = groups.group(0)
                    # remove trailing special character and whitespace
                    title = re.sub(
                        r"[,.\\<>?;:\'\"\[{\]}`~!@#$%^&*()\-_=+|\s]*$",
                        "",
                        title,
                    )
                    title = title.rstrip()
                    # check if title is empty after regex application and restore it in case
                    if not title:
                        title = title_before_regex
                        # remove trailing special character and whitespace
                        title = re.sub(
                            r"[,.\\<>?;:\'\"\[{\]}`~!@#$%^&*()\-_=+|\s]*$",
                            "",
                            title,
                        )
                        title = title.rstrip()
                    # define person metadata
                    person = metadata["person"]
                    # create list of queries to be performed based on available metadata
                    queries, combs = create_queries_list(title, person, year)
                    # perform the queries and stop when a match is returned
                    matched = None
                    exception = False
                    for i, query in enumerate(queries):
                        exception = False
                        params = {
                            "action": "query",
                            "format": "json",
                            "list": "search",
                            "srsearch": query
                        }

                        response = requests.get(wikidata_api_url, params=params)
                        response.raise_for_status()

                        data = response.json()

                        if "error" in data and data["error"]["info"].startswith("Search request is longer"):
                            # if the query is too long for wikidata, it could be due to a long person string or title
                            # in that case, we move at the next query
                            exception = True
                            continue

                        if "search" in data["query"] and data["query"]["search"]:
                            matched = analyze_result(data, correct_category_items, asin, combs[i])
                            if matched:
                                break

                    if matched is None:
                        if exception:
                            logging.info("%s - long-title-exception" % (asin,))
                        else:
                            logging.info("%s - not-found-query" % (asin,))
                    else:
                        if not matched:
                            logging.info("%s - not-in-correct-category" % (asin,))
                else:
                    logging.info("%s - not-title" % (asin,))

        except Exception:
            print(traceback.format_exc())
            logging.info("%s - exception" % (asin,))

    # use parallel computing to perform HTTP requests
    try:
        Parallel(n_jobs=n_cores, prefer="threads")(delayed(entity_link)(item) for item in m_data.items())
    except (KeyboardInterrupt, Exception):
        update_file(file_handler, temp_dict, mapping_file)
        print("Interruption occurred! Mapping file has been saved!")
        exit()

    update_file(file_handler, temp_dict, mapping_file)


def analyze_result(data: dict, correct_items: dict, asin: str, matched_using: str):
    for item in data["query"]["search"]:
        if item["title"] in correct_items["ids"]:
            logging.info("%s - %s - %s" % (asin, item["title"], matched_using))
            return True
    return False


def update_file(file_handler, temp_dict, mapping_file):
    # close the file handler
    file_handler.close()
    # create dictionary with new retrieved data
    with open('./output.log', 'r') as file:
        for line in file:
            split_line = line.split(" - ")
            if len(split_line) == 3:
                amazon_id, wiki_id, matching_attrs = split_line
                temp_dict[amazon_id] = {"wiki_id": wiki_id, "matching_attributes": matching_attrs.strip()}
            else:
                amazon_id, error = split_line
                temp_dict[amazon_id] = error.strip()
    # save to file - if the file was already existing, it will be updated. If not, it will be created
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(temp_dict, f, ensure_ascii=False, indent=4)
    # delete temporary log file
    os.remove("./output.log")


def create_queries_list(title, person, year):
    queries, combs = None, None
    if person is not None and year is not None:
        queries = [title + " " + person + " " + str(year), title + " " + person, title + " " + str(year), title]
        combs = ["title,person,year", "title,person", "title,year", "title"]
    elif person is not None and year is None:
        queries = [title + " " + person, title]
        combs = ["title,person", "title"]
    elif person is None and year is not None:
        queries = [title + " " + str(year), title]
        combs = ["title,year", "title"]
    else:
        queries = [title]
        combs = ["title"]

    return queries, combs

