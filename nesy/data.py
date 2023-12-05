import ast
import os
from joblib import Parallel, delayed
import json
from multiprocessing import Manager
import csv
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
import requests
from tqdm import tqdm
import Levenshtein
import re
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.chrome.options import Options
import random


def create_asin_metadata_json(metadata):
    """
    This function takes as input the JSON file containing Amazon product metadata and produces a filtered version of
    the same file, where for each ASIN we store information about the title.

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
            md_dict[row_dict["asin"]] = "no-title"

    with open(metadata) as f:
        Parallel(n_jobs=os.cpu_count())(delayed(process_line)(line, md_dict) for line in f)

    with open('./data/processed/%s' % (metadata.split("/")[-1]), 'w', encoding='utf-8') as f:
        json.dump(md_dict.copy(), f, ensure_ascii=False, indent=4)


def filter_metadata(metadata, rating_files):
    """
    This function filter the given metadata file using the item IDs found in the given rating files.
    It creates a new metadata file in the same location of metadata, called filtered-metadata.

    :param metadata: path to metadata file
    :param rating_files: list of rating file names. Provide only the name (e.g., reviews_Movies_and_TV_5)
    """
    desired_asin = []
    for dataset in rating_files:
        data = pd.read_csv("./data/processed/%s.csv" % (dataset, ))
        desired_asin.extend(data["itemId"].unique())

    with open(metadata) as json_file:
        m_data = json.load(json_file)

    new_file_dict = {k: m_data[k] for k in desired_asin}

    with open('./data/processed/filtered-%s' % (metadata.split("/")[-1]), 'w', encoding='utf-8') as f:
        json.dump(new_file_dict, f, ensure_ascii=False, indent=4)


def scrape_title(asins, bs=100, selenium=True):
    """
    This function takes as input a list of Amazon ASIN and performs an http request to get the title of the ASIN
    from the Amazon website.

    :param asins: list of ASIN for which the title has to be retrieved
    :param bs: batch size denoting the number of asin that has to be processed for each scraping job. Ideally,
    it should be len(asins) / #cpu
    :param selenium: whether to use selenium for scraping or classic HTTP requests
    :return: new dictionary containing key-value pairs with ASIN-title
    """
    managar = Manager()
    title_dict = managar.dict()
    if selenium:
        # Set up the Chrome options for a headless browser
        chrome_options = Options()
        chrome_options.add_argument('--headless=new')
        chrome_options.add_argument('--disable-gpu')  # Disable GPU to avoid issues in headless mode
        chrome_options.add_argument('--window-size=1920x1080')  # Set a window size to avoid responsive design

    def single_request(asins, title_dict):
        """
        This function performs a sigle HTTP request.

        :param asin: ASIN of the product for which the title has to be retrieved
        """
        urls = [f'https://www.amazon.com/dp/{asin}' for asin in asins]
        # urls = [f'https://camelcamelcamel.com/product/{asin}' for asin in asins]
        if selenium:
            # Set up the Chrome driver (you'll need to have ChromeDriver installed)
            chrome_service = ChromeService(executable_path='./chromedriver')
            driver = webdriver.Chrome(service=chrome_service, options=chrome_options)
        for url in urls:
            asin = url.split("/")[-1]
            try:
                if selenium:
                    # Load the Amazon product page
                    driver.get(url)
                    # Find the product title
                    title_element = driver.find_element(By.ID, 'productTitle')
                    if title_element:
                        title_dict[asin] = title_element.text.strip()
                    else:
                        title_dict[asin] = "no-title"
                    print(title_element.text.strip())
                else:
                    # Send a GET request to the Amazon product page
                    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) '
                                             'Chrome/54.0.2840.90 Safari/537.36'}
                    response = requests.get(url, headers=headers)
                    response.raise_for_status()
                    # Parse the HTML content of the page
                    soup = BeautifulSoup(response.text, 'html.parser')
                    # Find the product title
                    # title_element = soup.find('span', {'id': 'productTitle'})
                    title_element = soup.find('p', {'id': 'lowerpimg'})
                    print(title_element)
                    pd.d()
                    print(title_element.text.strip())
            except Exception as e:
                print(f"Error: {e}")
                title_dict[asin] = "no-title"

    Parallel(n_jobs=os.cpu_count())(delayed(single_request)(a, title_dict)
                                    for a in [asins[i:(i + bs if i + bs < len(asins) else len(asins))]
                                              for i in range(0, len(asins), bs)])
    return title_dict.copy()


def metadata_scraping(metadata):
    """
    This function takes as input a metadata file with some missing titles and uses web scraping to retrieve these
    titles from the Amazon website.

    At the end, a new metadata file with complete information is generated.

    :param metadata: file containing metadata for the items
    """
    with open(metadata) as json_file:
        m_data = json.load(json_file)
    no_titles = [k for k, v in m_data.items() if v == "no-title"]
    b = time.time()
    m_data = m_data | scrape_title(no_titles[:300], 30, selenium=True)
    print(time.time() - b)
    with open('./data/processed/complete-%s' % (metadata.split("/")[-1]), 'w', encoding='utf-8') as f:
        json.dump(m_data, f, ensure_ascii=False, indent=4)


def create_pandas_dataset(data):
    """
    This function takes as input the JSON file containing Amazon reviews and produces a CSV file. Each record has
    user ID, item ID, rating, timestamp.

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
        :param rating_dict: list containing ratings to be saved in the CSV file. Each rating is a dict.
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


# this function is too slow due to the filtering by title with the REGEX
def entity_linker_sparql(dataset, metadata):
    """
    It creates a mapping between Amazon ASIN and Wikidata IDs

    :param dataset: path to the processed Amazon dataset containing ratings for users on items
    :param metadata: path to the processed Amazon metadata
    :return: CSV file containing the mapping between the given dataset and wikidata
    """
    # read data
    data = pd.read_csv(dataset)
    # get item IDs
    item_ids = data['itemId'].unique()
    # read metadata
    with open(metadata) as json_file:
        m_data = json.load(json_file)

    def entity_link(asin, category, match_dict):
        """
        It performs a sparql query over wikidata to retrieve the wikidata ID of the given item.

        :param asin: asin of the Amazon item for which the ID is requested
        :param category: whether the item is a movie, music, or book
        :param match_dict: python dictionary containing the matches. It contains an empty string when no match is found
        on wikidata or the query went out of time.
        """
        query = """
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                SELECT DISTINCT ?item
                WHERE {
                    ?item wdt:P31/wdt:P279* %s .
                    ?item rdfs:label|skos:altLabel ?label .
                    FILTER regex(?label, "%s", "i") . 
                }
        """ % (category, m_data[asin]["title"].replace(" ", ".*"))
        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        sparql.setReturnFormat(JSON)
        sparql.setQuery(query)

        try:
            ret = sparql.queryAndConvert()
            match_dict[asin] = [v for r in ret["results"]["bindings"] for item in r for t, v in item]
            print("match")
        except:
            match_dict[asin] = []
            print("no match")

    # here the parallel computing of sparql queries
    manager = Manager()
    match_dict = manager.dict()
    category = "wd:Q11424" if "Movies" in dataset else ("wd:Q7725634" if "Books" in dataset else "wd:Q482994")
    Parallel(n_jobs=os.cpu_count())(delayed(entity_link)(asin, category, match_dict) for asin in item_ids)
    match_dict = match_dict.copy()
    return match_dict


# too slow
def get_wid_title(wids):
    """
    It takes as input a JSON file containing the URIs of some wikidata items and creates a JSON file containing
    a mapping from Wikidata id to item title.

    :param wids: path to JSON file
    """
    # read JSON
    with open(wids) as json_file:
        wids_ = json.load(json_file)
    wids_ = [v["value"].split("/")[-1] for obj in wids_["results"]["bindings"] for t, v in obj.items()]
    w_links = ["https://www.wikidata.org/wiki/Special:EntityData/%s.json" % (id_, ) for id_ in wids_]
    headers = {'Accept': 'application/json'}

    def get_json(link, wid_title_dict):
        """
        It retrieves the JSON file at the given link using an HTTP request.
        It also takes the entity title and aliases and save them in a dictionary.

        :param link: link to the JSON file
        """
        json_file = requests.get("https://www.wikidata.org/w/api.php?action=wbgetentities&ids=Q459173%7CQ544%7CQ3241540%7CQ185969%7CQ715269%7CQ27527&languages=en&props=aliases&format=json", headers=headers).json()
        wid = link.split("/")[-1].split(".")[0]
        wid_title_dict[wid] = {"labels": json_file["entities"][wid]["labels"],
                               "aliases": json_file["entities"][wid]["aliases"]}
    # parallel computing
    manager = Manager()
    wid_title_dict = manager.dict()
    Parallel(n_jobs=1)(delayed(get_json)(link, wid_title_dict) for link in tqdm(w_links))
    wid_title_dict = wid_title_dict.copy()
    with open('./data/processed/%s' % (wids.split("/")[-1], ), 'w', encoding='utf-8') as f:
        json.dump(wid_title_dict, f, ensure_ascii=False, indent=4)


# questo e' stato abbastanza veloce
def get_wid_labels(wids):
    """
    It takes as input a JSON file containing the URIs of some wikidata items and creates a JSON file containing
    a mapping from Wikidata id to item labels and aliases.

    :param wids: path to JSON file
    """
    # read JSON
    with open(wids) as json_file:
        wids_ = json.load(json_file)
    wids_ = [v["value"].split("/")[-1] for obj in wids_["results"]["bindings"] for t, v in obj.items()]
    headers = {'Accept': 'application/json'}

    def get_json(idx, wid_title_dict):
        """
        It retrieves the JSON files for 50 items starting from the given index by using an HTTP request to the
        wikidata API.
        It also takes the labels and aliases for all the 50 entities and save them in a dictionary.

        :param idx: idx of the first of 50 items for which the metadata has to be taken
        :param wid_title_dict: dict where the information have to be stored
        """
        end_idx = min(idx + 50, len(wids_))
        response = requests.get(
            "https://www.wikidata.org/w/api.php?action=wbgetentities&ids=%s&languages=en&props=labels|aliases&format=json" %
            ("|".join(wids_[idx:end_idx]),),
            headers=headers)
        try:
            response.raise_for_status()
            json_file = response.json()
            wid_title_dict.update({wid: {"aliases": json_file["entities"][wid]["aliases"],
                                         "labels": json_file["entities"][wid]["labels"]}
                                   for wid in json_file["entities"]
                                   if "labels" in json_file["entities"][wid]
                                   if "aliases" in json_file["entities"][wid]})
        except Exception as e:
            # this is printed for debugging
            print(e)

    # parallel computing
    manager = Manager()
    wid_title_dict = manager.dict()
    Parallel(n_jobs=os.cpu_count())(delayed(get_json)(idx, wid_title_dict) for idx in tqdm(range(0, len(wids_), 50)))
    with open('./data/processed/wikidata-%s' % (wids.split("/")[-1],), 'w', encoding='utf-8') as f:
        json.dump(wid_title_dict.copy(), f, ensure_ascii=False, indent=4)


# even in parallel, it is too slow, so it is not good
def entity_linker_local(amazon_ratings):
    """
    This function is similar to the function entity_linker_sparql but it works locally, with a created dump of
    wikidata entities for which labels are aliases are saved. The linking is purely based on string pattern matching
    between the item's title saved on the Amazon metadata and the wikidata's labels and aliases saved locally.

    The function creates a mapping and produces a file containing this mapping.

    :param amazon_ratings: pandas dataframe containing rating on Amazon items
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

    def sim(str1, str2, regex=False):
        """
        It computes the similarity between two given strings. It uses the Levenshtein distance to compute the similarity
        between the strings.

        :param str1: first string
        :param str2: second string
        :param regex: if True, pattern matching using regex is used in place of Levenshtein distance
        :return: the similarity score between the two strings
        """
        if regex:
            pattern = re.compile(r'%s' % (str1.replace(" ", ".*"), ), re.IGNORECASE)
            if pattern.search(str2):
                return 1
            else:
                return 0
        else:
            distance = Levenshtein.distance(str1, str2)
            return 1 - distance / max(len(str1), len(str2))

    def entity_link(asin, match_dict):
        """
        It performs a search over the local wikidata to retrieve the wikidata ID of the given item ASIN.

        :param asin: asin of the Amazon item for which the ID is requested
        :param match_dict: python dictionary containing the matches. It contains an empty string when no match is found
        on wikidata or the query went out of time.
        """
        amazon_title = m_data[asin]["title"]
        for wid in wikidata_dump:
            # see if it matches the label
            try:
                if "en" in wikidata_dump[wid]["labels"]:
                    if sim(amazon_title, wikidata_dump[wid]["labels"]["en"]["value"], regex=True):
                        if asin in match_dict:
                            match_dict[asin].append(wid)
                        else:
                            match_dict[asin] = [wid]
                    else:
                        # see if it matches one alias
                        if "en" in wikidata_dump[wid]["aliases"]:
                            for alias in wikidata_dump[wid]["aliases"]["en"]:
                                if sim(amazon_title, alias["value"]):
                                    if asin in match_dict:
                                        match_dict[asin].append(wid)
                                    else:
                                        match_dict[asin] = [wid]
                                    break
            except Exception as error:
                print(error)
                print(wikidata_dump[wid])

    # here the parallel computing of sparql queries
    manager = Manager()
    match_dict = manager.dict()
    Parallel(n_jobs=os.cpu_count())(delayed(entity_link)(asin, match_dict) for asin in tqdm(item_ids))
    match_dict = match_dict.copy()
    with open('./data/processed/mapping-%s' % (amazon_ratings.split("/")[-1], ), 'w', encoding='utf-8') as f:
        json.dump(match_dict, f, ensure_ascii=False, indent=4)


# this works well
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


# this works well
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
