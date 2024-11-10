import sys
from SPARQLWrapper import SPARQLWrapper, JSON
import json


def get_ids(category: str) -> None:
    """
    This function executes a SPARQL query on the Wikidata Query Service to get the Wikidata ID of all the entities that
    are instance of subclasses of the given category.

    This list of IDs can then be used by the entity linker to check if a matched item belongs to the desired category.
    If it belongs to the category, the match is confirmed. If it does not belong to the category, the match is
    discarded.

    It creates a JSON file containing the list of Wikidata IDs retrieved with the query.

    :param category: string representing the category. "movies", "music", or "books" are the possible categories. When
    the category is "movies", it looks for instances of subclasses of moving image, when the category is "music", it
    looks for instances of subclasses of musical work, when the category is "books", it looks for instances of
    subclasses of written work.
    """
    # set the URL to the Wikidata Query Service
    endpoint_url = "https://query.wikidata.org/sparql"

    # create the correct query based on the given category
    if category == "movies":
        query = """SELECT DISTINCT ?item
                   WHERE {
                        ?item wdt:P31/wdt:P279* wd:Q10301427.
                   }
                   """
    elif category == "music":
        query = """SELECT DISTINCT ?item
                    WHERE {
                        ?item wdt:P31/wdt:P279* wd:Q2188189.
                    }"""
    elif category == "books":
        query = """SELECT DISTINCT ?item
                    WHERE {
                        ?item wdt:P31/wdt:P279* wd:Q7725634.
                        MINUS {
                            ?item wdt:P31/wdt:P279* wd:Q7366.
                        }
                    }
                    """
    else:
        raise Exception("A wrong category has been given!")

    # create dict containing the results
    retrieved_ids = {"ids": []}

    # execute the query
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    for result in results["results"]["bindings"]:
        retrieved_ids["ids"].append(result["item"]["value"].split("/")[-1])

    # save Wikidata IDs to file
    with open('./data/processed/ids_in_categories/wikidata-%s.json' % (category,), 'w',
              encoding='utf-8') as f:
        json.dump(retrieved_ids, f, ensure_ascii=False, indent=4)


def linking_stats(mapping_file, errors, save_asins=False):
    """
    This function produces some statistics for the provided mapping file. The statistics include the number of items
    that have been correctly matched on wikidata plus some statistics regarding the errors occurred while matching.

    The parameter `errors` allows to define which statistics to include in the output, chosen from (not-title,
    exception, not-found-query, not-in-correct-category).

    If save_asins is set to True, for each of these statistics, the output will contain the set of ASINs corresponding
    to each error.

    The output will be saved to a JSON file in the same location of the given metadata file with "_stats" added at
    the name. This, if save_asins is set to True, otherwise it just prints on the standard output.

    :param mapping_file: path to the mapping file containing the matches between Amazon and Wikidata
    :param errors: list of strings containing the name of the errors that have to be included in the statistics. The
    script will search for these specific names in the values of the metadata, for each of the ASINs
    :param save_asins: whether to save the set of the ASINs for each statistic or just produce the statistic
    """
    errors = {e: {"counter": 0, "asins": []} for e in errors}
    with open(mapping_file) as json_file:
        mapping = json.load(json_file)
    for asin, data in mapping.items():
        if type(data) is not dict:
            if data in errors:
                errors[data]["counter"] += 1
                if save_asins:
                    errors[data]["asins"].append(asin)
        else:
            if "matched" in errors:
                errors["matched"]["counter"] += 1
            else:
                errors["matched"] = {"counter": 1}

    if save_asins:
        with open(mapping_file[:-5] + "_stats.json", 'w', encoding='utf-8') as f:
            json.dump(errors, f, ensure_ascii=False, indent=4)
    else:
        print(errors)
