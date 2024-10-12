import itertools
import json
import logging
import traceback

from joblib import delayed
from neo4j import GraphDatabase, ManagedTransaction
from neo4j.exceptions import ClientError, CypherSyntaxError

from nesy.utils import ParallelTqdm
from .utils import (
    get_cold_start,
    get_popular,
    get_rating_stats,
    refine_cold_start_items,
    refine_popular_items,
)


def neo4j_path_finder(
    mapping_file_1: str,
    mapping_file_2: str,
    max_hops: int = 2,
    shortest_path: bool = True,
    cold_start: bool = False,
    popular: bool = False,
    cs_threshold: int = 5,
    pop_threshold: int = 50,
    n_cores: int = 1,
) -> None:
    """
    This function computes all the available Wikidata's paths between matched entities in mapping_file_1 and
    matched entities in mapping_file_2. It saves all these paths in a JSON file.

    :param mapping_file_1: first mapping file
    :param mapping_file_2: second mapping file
    :param max_hops: maximum number of hops allowed for the path
    :param shortest_path: whether to find just the shortest path or all the paths connecting the two entities
    :param cold_start: whether to compute paths just for cold-start items in the target domain
    :param popular: whether to compute paths just for popular items in the source domain
    :param cs_threshold: threshold to select cold-start items
    :param pop_threshold: threshold to select popular items
    :param n_cores: number of processors to be used to execute this function
    """
    # read mapping files
    with open(mapping_file_1) as json_file:
        m_1 = json.load(json_file)
        if popular:
            if "movies" in mapping_file_1:
                path = "./data/processed/legacy/reviews_Movies_and_TV_5.csv"
            elif "music" in mapping_file_1:
                path = "./data/processed/legacy/reviews_CDs_and_Vinyl_5.csv"
            else:
                path = "./data/processed/legacy/reviews_Books_5.csv"
            stats = get_rating_stats(path, "item")
            pop = get_popular(stats, pop_threshold)
            m_1 = refine_popular_items(pop, mapping_file_1)
    with open(mapping_file_2) as json_file:
        m_2 = json.load(json_file)
        if cold_start:
            if "movies" in mapping_file_2:
                path = "./data/processed/legacy/reviews_Movies_and_TV_5.csv"
            elif "music" in mapping_file_2:
                path = "./data/processed/legacy/reviews_CDs_and_Vinyl_5.csv"
            else:
                path = "./data/processed/legacy/reviews_Books_5.csv"
            stats = get_rating_stats(path, "item")
            cs = get_cold_start(stats, cs_threshold)
            m_2 = refine_cold_start_items(cs, mapping_file_2)
    # create logger for logging everything to file in case the long executions are interrupted
    # Configure the logger
    logging.basicConfig(level=logging.INFO)  # Set the desired log level
    # Remove all handlers to avoid printing on stdout
    for handler in logging.getLogger().handlers[:]:
        logging.getLogger().removeHandler(handler)
    # Create a FileHandler to write log messages to a file
    file_handler = logging.FileHandler("output.log")
    # Add the file handler to the logger
    logging.getLogger().addHandler(file_handler)
    # Define connection details for Neo4j
    uri = "bolt://localhost:7687"
    # Initialize the driver
    driver = GraphDatabase.driver(uri)
    # create query template given the maximum number of hops
    query = get_query(max_hops, shortest_path)

    def find_path(pair: tuple) -> None:
        """
        It performs a query to find the paths between the given pair of items on Neo4j.

        :param pair: tuple containing the ids on which the query has to be performed
        """
        first_item, second_item = pair
        first_item_asin, first_item_wiki_id = first_item
        second_item_asin, second_item_wiki_id = second_item
        try:
            # execute query
            with driver.session() as session:
                session.execute_write(
                    execute_query, query, first_item_wiki_id, second_item_wiki_id
                )
        except (CypherSyntaxError, ClientError) as e:
            print(e)
            exit(1)
        except Exception:
            print(traceback.format_exc())
            logging.info(
                "%s -/- %s -/- exception -/- 0" % (first_item_asin, second_item_asin)
            )

    # computing total number of tasks
    total_tasks = compute_n_tasks(m_1, m_2)
    # get pairs for which the paths have to be generated
    pairs = get_pairs(m_1, m_2)
    # use parallel computing to perform queries
    try:
        ParallelTqdm(n_jobs=n_cores, prefer="threads", total_tasks=total_tasks)(
            delayed(find_path)(pair) for pair in pairs
        )
    except (KeyboardInterrupt, Exception):
        print(traceback.format_exc())
        print("Interruption occurred! Path file has been saved!")
        exit(0)


def get_query(max_hops: int, shortest_path: bool) -> str:
    """
    This function creates the query for Neo4j based on the given number of hops.

    :param max_hops: max number of hops allowed for the path
    :param shortest_path: whether to configure the query to find just the shortest path or not
    :return: the query to be executed
    """
    query_head = "(n1:entity {wikidata_id: $first_item})"
    query_tail = "(n2:entity {wikidata_id: $second_item})"
    if not shortest_path:
        query = ""
        for i in range(max_hops):
            query += "MATCH path=%s" % (query_head,)
            for j in range(i):
                query += "-[*1..1]-(mid%d:entity)" % (j + 1,)
            query += "-[*1..1]-"
            query += "%s RETURN path, length(path) AS path_length" % (query_tail,)
            if i != max_hops - 1:
                query += " UNION "
    else:
        query = f"""
            OPTIONAL MATCH {query_head}-[r:SPECIAL_PATH]-{query_tail}
            WITH COUNT(r) > 0 AS pathExists
            
            WHERE pathExists = false
            MATCH path = shortestPath({query_head}-[r:relation*1..{max_hops}]-{query_tail})
            WITH path, length(path) AS pathLength, n1, n2,
                 REDUCE(s = "", n IN nodes(path) | 
                    s + 
                    CASE 
                        WHEN size(s) > 0 THEN " -> " 
                        ELSE "" 
                    END + 
                    n.wikidata_id
                 ) AS path_string
            MERGE (n1)-[r1:SPECIAL_PATH {{path_length: pathLength, path_string: path_string}}]-(n2)
            MERGE (n2)-[r2:SPECIAL_PATH {{path_length: pathLength, path_string: path_string}}]-(n1)
            """
    return query


def execute_query(
    tx: ManagedTransaction, query: str, first_item: str, second_item: str
):
    """
    This function executes the given query on Neo4j.
    :param tx:
    :param query: the query to be executed
    :param first_item: first query parameter value
    :param second_item: second query parameter value
    :return: the results of the query
    """
    tx.run(query, first_item=first_item, second_item=second_item)  # type: ignore


def compute_n_tasks(mapping_1: dict, mapping_2: dict = None) -> int:
    """
    This function computes the total number of pairs for which the paths have to be computed.

    :param mapping_1: dictionary containing the mapping between Amazon and wikidata in the source domain
    :param mapping_2: dictionary containing the mapping between Amazon and wikidata in the target domain
    :return: number of pairs for which the paths have to be computed
    """
    matched_items_1, matched_items_2 = 0, 0
    if isinstance(mapping_1, list):
        matched_items_1 = len(mapping_1)
    else:
        for item in mapping_1:
            if isinstance(mapping_1[item], dict):
                matched_items_1 += 1
    if mapping_2 is not None:
        if isinstance(mapping_2, list):
            matched_items_2 = len(mapping_2)
        else:
            for item in mapping_2:
                if isinstance(mapping_2[item], dict):
                    matched_items_2 += 1
        return matched_items_1 * matched_items_2
    else:
        return matched_items_1


def get_pairs(source_d: dict, target_d: dict):
    """
    This function computes the pairs for which the paths have to be generated and return a generator of pairs.

    :param source_d: source domain dict
    :param target_d: target domain dict
    :return: pairs for which the paths have to be generated returned as a generator
    """
    if not isinstance(source_d, list):
        source_d_ids = [
            (asin, data["wiki_id"])
            for asin, data in source_d.items()
            if isinstance(source_d[asin], dict)
        ]
    else:
        source_d_ids = source_d
    if not isinstance(target_d, list):
        target_d_ids = [
            (asin, data["wiki_id"])
            for asin, data in target_d.items()
            if isinstance(target_d[asin], dict)
        ]
    else:
        target_d_ids = target_d
    return itertools.product(source_d_ids, target_d_ids)
