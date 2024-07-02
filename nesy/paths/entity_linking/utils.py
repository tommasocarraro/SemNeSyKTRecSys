import sys
from SPARQLWrapper import SPARQLWrapper, JSON
import json
from SPARQLWrapper.SPARQLExceptions import EndPointInternalError
import time


def get_ids(category: str) -> None:
    """
    This function executes a SPARQL query on the Wikidata Query Service to get the Wikidata ID of all the entities that
    are instance of subclasses of the given category.

    This list of IDs can then be used by the entity linker to check if a matched item belongs to the desired category.
    If it belongs to the category, the match is confirmed. If it does not belong to the category, the match is
    discarded.

    It creates a JSON file containing the list of Wikidata IDs retrieved with the query.

    :param category: string representing the category. "movies", "music", or "books" are the possible categories. When
    the category is "movies", it looks for instances of subclasses of audiovisual work, when the category is "music", it
    looks for instances of subclasses of musical work, when the category is "books", it looks for instances of
    subclasses of written work.
    """
    # set the URL to the Wikidata Query Service
    endpoint_url = "https://query.wikidata.org/sparql"

    # create the correct query based on the given category
    if category == "movies":
        query = """SELECT DISTINCT ?item
                   WHERE {
                        ?item wdt:P31/wdt:P279* wd:Q2431196.
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
    with open('./data/processed/legacy/wikidata-%s.json' % (category,), 'w',
              encoding='utf-8') as f:
        json.dump(retrieved_ids, f, ensure_ascii=False, indent=4)
