import ast
import os
from joblib import Parallel, delayed
import json
from multiprocessing import Manager
import csv
import pandas as pd
import requests
import re
from concurrent.futures import ThreadPoolExecutor
import sys
from SPARQLWrapper import SPARQLWrapper, JSON
import logging
from tqdm import tqdm


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


def entity_linker_api_query(amazon_ratings, use_dump=True, retry=False, retry_reason=None, add_dump=False):
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
    :param add_dump: whether in case of music mapping an additional dump with movie wiki IDs has to be used. This is
    useful since a lot of albums in Amazon are just DVD of the albums. Defaults to False.
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
                            if "CDs" in amazon_ratings and add_dump:
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


def entity_linker_title_person_year(amazon_metadata, retry=False, parallel=False):
    """
    This function uses the Wikidata API (action=query) to get the ID of wikidata items corresponding to the Amazon
    items in the dataset. The API is accessed using HTTP requests. It takes as input a JSON file containing metadata on
    Amazon items. This metadata contains title, person (i.e., director in case of movie, artist in case of song, and
    writer in case of books), and publication year for the item. The title of the Amazon item is used for querying
    Wikidata Special Search. This function first iterates through the found items and filter our items which are not in
    the correct category. For example, if we are looking for a book and the result contains some movies, these are
    filter out. Then, it gets the wikidata information for all the other items and look for person and publication year.
    If the person and publication year match, then the match is saved.

    At the end, a new JSON file is created. The keys will be the ASINs and the values the wikidata IDs.

    :param amazon_metadata: JSON file containing metadata on Amazon items
    :param retry: valid from second execution. If this is set to true, the script will retry to find matches for all
    the items that during the first search did not get a match due to an error.
    :param parallel: whether to use multiple processors for executing this function
    """
    # read metadata
    with open(amazon_metadata) as json_file:
        m_data = json.load(json_file)
    # load wikidata dumps
    wiki_dumps = {}
    wikidata_movies = "./data/processed/legacy/wikidata-movies.json"
    wikidata_books = "./data/processed/legacy/wikidata-books.json"
    wikidata_music = "./data/processed/legacy/wikidata-music.json"
    with open(wikidata_music) as json_file:
        wiki_dumps["music"] = json.load(json_file)
    with open(wikidata_books)as json_file:
        wiki_dumps["book"] = json.load(json_file)
    with open(wikidata_movies) as json_file:
        wiki_dumps["movie"] = json.load(json_file)
    # link to API
    wikidata_api_url = "https://www.wikidata.org/w/api.php"

    def entity_link(asin):
        """
        It performs an HTTP request on the Wikidata API to get the Wikidata ID of the given item (referred by ASIN).

        :param asin: asin of the Amazon item for which the ID is requested

        It returns a tuple where the first element is the ASIN and the second element is the Wikidata ID or a string
        indicating the reason why the match could not be found.
        """
        try:
            # check if we have the title for the item
            if m_data[asin]["title"] is not None:
                # remove punctuation and strange parenthesis
                # todo capire se ha senso sta cosa
                amazon_title = re.sub("[\(\[].*?[\)\]]", "", m_data[asin]["title"])
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
                    for item in data["query"]["search"]:
                        if item["title"] in wiki_dumps[m_data[asin]["type"]]["wids"]:
                            # we need to open the wikidata page of this item
                            entity_url = f'https://www.wikidata.org/wiki/Special:EntityData/{item["title"]}.json'
                            entity_response = requests.get(entity_url)
                            entity_data = entity_response.json()
                            # Extract relevant data from the page
                            claims = entity_data['entities'][item["title"]]['claims']

                            person_match = None
                            if m_data[asin]["person"] is not None:
                                if m_data[asin]["type"] == "movie":
                                    person_claims = claims.get('P57', [])
                                elif m_data[asin]["type"] == "music":
                                    person_claims = claims.get('P175', [])
                                else:
                                    person_claims = claims.get('P50', [])

                                # get person wikidata ID
                                params = {
                                    "action": "query",
                                    "format": "json",
                                    "list": "search",
                                    "srsearch": m_data[asin]["person"]
                                }

                                response = requests.get(wikidata_api_url, params=params)
                                response.raise_for_status()

                                data = response.json()
                                person_id = None
                                if "search" in data["query"] and data["query"]["search"]:
                                    person_id = data["query"]["search"][0]["title"]
                                person_match = any(
                                    claim['mainsnak']['datavalue']['value']['id'] == person_id
                                    for claim in person_claims
                                )

                            release_date_match = None
                            if m_data[asin]["year"] is not None:
                                release_date_claims = claims.get('P577', [])
                                release_date_match = any(
                                    claim['mainsnak']['datavalue']['value']['time'][1:5] == str(m_data[asin]["year"])
                                    for claim in release_date_claims
                                )
                            # todo gestire tutte le casistiche qui, casi in cui non c'e' regista o titolo
                            # todo sistemare il type dove manca sui metadati
                            if person_match and release_date_match:
                                print("%s - %s - %s" % (asin, m_data[asin], item["title"]))
                                return asin, item["title"]

                    print("%s not found in dump" % (asin,))
                    return asin, "not-in-dump"  # all found items are not of the correct category
                else:
                    print("%s not found by query" % (asin,))
                    return asin, "not-found-query"  # there are no results for the query
            else:
                print("%s does not have a title" % (asin,))
                return asin, "no-title"  # the item has not a corresponding title in the metadata file
        except Exception as e:
            print("%s produced the exception %s" % (asin, e))
            return asin, "exception"  # the search produced an exception

    # here the parallel computing
    if parallel:
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            matches = list(executor.map(entity_link, list(m_data.keys())))
    else:
        matches = [entity_link(asin) for asin in list(m_data.keys())]

    match_dict = {k: v for k, v in matches}
    print(match_dict)

    # with open('./data/processed/mapping-%s.json' % (amazon_ratings.split("/")[-1].split(".")[0],), 'w',
    #           encoding='utf-8') as f:
    #     json.dump(match_dict, f, ensure_ascii=False, indent=4)


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


def get_mapping_map(dom):
    """
    This function returns the path to the mapping file corresponding to the given domain.

    :param dom: str indicating the domain name (movies, music, books)
    :return: path to the requested file
    """
    if dom == "movies":
        return "./data/processed/mapping-reviews_Movies_and_TV_5.json"
    if dom == "music":
        return "./data/processed/mapping-reviews_CDs_and_Vinyl_5.json"
    if dom == "books":
        return "./data/processed/mapping-reviews_Books_5.json"


def get_wiki_ids(mapping_dict):
    """
    This function returns the list of wikidata IDs corresponding to the given mapping dictionary.

    :param mapping_dict: dictionary containing dataset_id:wikidata_id pairs
    :return: list of wikidata IDs
    """
    return [id_ for id_ in list(mapping_dict.values()) if id_.startswith("Q")]


def get_cross_pairs(dom1, dom2):
    """
    This function returns a list of all the possible combinations of the wikidata IDs in the two given domains.

    :param dom1: str indicating the first domain.
    :param dom2: str indicating the second domain.
    :return: a generator of all combinations of items in the two different domains.
             the length of the generator.
    """
    # get mapping file paths
    dom1 = get_mapping_map(dom1)
    dom2 = get_mapping_map(dom2)
    # get wikidata ID lists
    with open(dom1) as json_file:
        dom1 = get_wiki_ids(json.load(json_file))
    with open(dom2) as json_file:
        dom2 = get_wiki_ids(json.load(json_file))
    # return the cartesian product of the two sets
    return ((id1, id2) for id1 in dom1 for id2 in dom2), len(dom1) * len(dom2)


def remove_movies_from_music():
    """
    This function takes the mapping between Amazon CDs and wikidata and remove all the matched movies in this mapping.
    It creates a new JSON file without the mapped movies. Instead of the Wikidata ID, there will be the "not-in-dump"
    string.
    """
    with open("./data/processed/mapping-reviews_CDs_and_Vinyl_5.json") as json_file:
        cds = json.load(json_file)
    with open("./data/processed/wikidata-movies.json") as json_file:
        movies = json.load(json_file)
    new_cds = {amaz_id: wiki_id if wiki_id not in movies["wids"] else "not-in-dump" for amaz_id, wiki_id in cds.items()}

    with open('./data/processed/mapping-reviews_CDs_and_Vinyl_5_only_music.json', 'w', encoding='utf-8') as f:
        json.dump(new_cds, f, ensure_ascii=False, indent=4)


def split_metadata(metadata_path):
    """
    This function takes as input a full metadata file and split it into three files (movies, books, and music).

    It generates the new metadata file, one for each domain.

    :param metadata_path: path to the metadata file to be split
    """
    with open(metadata_path) as json_file:
        complete_metadata = json.load(json_file)
    path_prefix = "./data/processed/mapping-reviews_"
    for mapping in ["Books_5", "CDs_and_Vinyl_5_only_music", "Movies_and_TV_5"]:
        path = path_prefix + mapping + ".json"
        with open(path) as json_file:
            mapping_file = json.load(json_file)
            correct_asins = set(mapping_file.keys())
            new_dict = {asin: title for asin, title in complete_metadata.items() if asin in correct_asins}
            with open("./data/processed/" + mapping + "_metadata.json", 'w', encoding='utf-8') as f:
                json.dump(new_dict, f, ensure_ascii=False, indent=4)


def update_metadata(metadata_path, new_data):
    """
    Updates the given metadata with the given new data (ASIN is the key and a dict containing the title and other info
    is the value).

    :param metadata_path: path to metadata with ASIN - title pairs
    :param new_data: path to new data
    """
    # open metadata
    with open(metadata_path) as json_file:
        metadata = json.load(json_file)
    # open new data
    with open(new_data) as json_file:
        new_data = json.load(json_file)
    # update metadata
    metadata.update({k: v["title"] for k, v in new_data.items() if v["title"] is not None})
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
