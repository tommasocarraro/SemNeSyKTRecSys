import itertools
import json
from typing import Any, Callable

from joblib import delayed
from loguru import logger
from neo4j import GraphDatabase, ManagedTransaction
from neo4j.exceptions import ClientError, CypherSyntaxError

from config import NEO4J_DBNAME, NEO4J_URI
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
    n_threads: int = 1,
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
    :param n_threads: number of processors to be used to execute this function
    """

    # read mapping files
    def get_mapping_path(mapping_file_name: str):
        if "movies" in mapping_file_name:
            file_path = "./data/processed/legacy/reviews_Movies_and_TV_5.csv"
        elif "music" in mapping_file_name:
            file_path = "./data/processed/legacy/reviews_CDs_and_Vinyl_5.csv"
        elif "books" in mapping_file_name:
            file_path = "./data/processed/legacy/reviews_Books_5.csv"
        else:
            logger.error("Wrong mapping file name was supplied")
            exit(1)
        return file_path

    if popular:
        path = get_mapping_path(mapping_file_1)
        stats = get_rating_stats(path, "item")
        pop = get_popular(stats, pop_threshold)
        m_1 = refine_popular_items(pop, mapping_file_1)
    else:
        with open(mapping_file_1, "r") as json_file:
            m_1 = json.load(json_file)

    if cold_start:
        path = get_mapping_path(mapping_file_2)
        stats = get_rating_stats(path, "item")
        cs = get_cold_start(stats, cs_threshold)
        m_2 = refine_cold_start_items(cs, mapping_file_2)
    else:
        with open(mapping_file_2, "r") as json_file:
            m_2 = json.load(json_file)

    # Initialize the driver
    with GraphDatabase.driver(NEO4J_URI, max_connection_pool_size=n_threads) as driver:
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
                with driver.session(database=NEO4J_DBNAME) as session:
                    session.execute_write(
                        execute_query, query, first_item_wiki_id, second_item_wiki_id
                    )
            except (CypherSyntaxError, ClientError) as ex:
                logger.error(ex)
                exit(1)
            except Exception as ex:
                logger.error(ex)

        # computing total number of tasks
        total_tasks = compute_n_tasks(m_1, m_2)
        # get pairs for which the paths have to be generated
        pairs = get_pairs(m_1, m_2)
        # use parallel computing to perform queries
        try:
            ParallelTqdm(n_jobs=n_threads, prefer="threads", total_tasks=total_tasks)(
                delayed(find_path)(pair) for pair in pairs
            )
        except KeyboardInterrupt:
            logger.info("Manual interrupt detected. Gracefully quitting")
            driver.close()  # TODO: currently not graceful, needs work
            exit(0)
        except Exception as e:
            logger.error(e)
            exit(1)


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
        # this approach only works for shortest paths as it computes only a single path per pair
        # first it checks if the special path already exists, if it doesn't then it is computed
        # currently the path string is equal for both directions
        query = f"""
            OPTIONAL MATCH {query_head}-[r:SPECIAL_PATH]-{query_tail}
            WITH COUNT(r) > 0 AS pathExists
            
            WHERE pathExists = false
            MATCH p=shortestPath({query_head}-[r:relation*1..{max_hops}]-{query_tail})
            WITH p, n1, n2, 
                [n IN nodes(p) | n] AS nodeList,
                [r IN relationships(p) | r] AS relList
            WITH p, n1, n2, nodeList, relList,
                range(0, size(relList)-1) AS indexes
            WITH p, n1, n2, reduce(s = "", i IN indexes |
                s + 
                case 
                    when i = 0 
                    then "(" + coalesce(nodeList[0].wikidata_id, "") + ")"
                    else "" 
                end +
                case 
                    when startNode(relList[i]) = nodeList[i] 
                    then "-"
                    else "<-"
                end +
                "[" + coalesce(relList[i].wikidata_id, "") + "]" +
                case 
                    when endNode(relList[i]) = nodeList[i+1] 
                    then "->"
                    else "-"
                end +
                "(" + coalesce(nodeList[i+1].wikidata_id, "") + ")"
            ) AS pathString
            CREATE (n1)-[r1:SPECIAL_PATH {{path_length: length(p), path_string: pathString}}]->(n2)
            CREATE (n2)-[r2:SPECIAL_PATH {{path_length: length(p), path_string: pathString}}]->(n1)
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


def compute_n_tasks(
    mapping_1: list | dict[str, Any], mapping_2: list | dict[str, Any] = None
) -> int:
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


def get_pairs(source_d: list | dict[str, Any], target_d: list | dict[str, Any]):
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
