import logging


def entity_linker_api_query(amazon_metadata, use_dump=True, retry=False, retry_reason=None, add_dump=False):
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
