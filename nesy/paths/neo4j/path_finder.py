from itertools import product

import orjson
from joblib import delayed
from loguru import logger
from neo4j import GraphDatabase
from neo4j.exceptions import DriverError, Neo4jError

from config import NEO4J_DBNAME, NEO4J_PASS, NEO4J_URI, NEO4J_USER
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

    def get_mapping_domain(mapping_file_name: str):
        if "movies" in mapping_file_name:
            return "'movies'"
        elif "music" in mapping_file_name:
            return "'music'"
        elif "books" in mapping_file_name:
            return "'books'"
        else:
            logger.error("Wrong mapping file name was supplied")
            exit(1)

    if popular:
        path = get_mapping_path(mapping_file_1)
        stats = get_rating_stats(path, "item")
        pop = get_popular(stats, pop_threshold)
        m_1 = refine_popular_items(pop, mapping_file_1)
    else:
        with open(mapping_file_1, "rb") as json_file:
            m_1 = orjson.loads(json_file.read())

    if cold_start:
        path = get_mapping_path(mapping_file_2)
        stats = get_rating_stats(path, "item")
        cs = get_cold_start(stats, cs_threshold)
        m_2 = refine_cold_start_items(cs, mapping_file_2)
    else:
        with open(mapping_file_2, "rb") as json_file:
            m_2 = orjson.loads(json_file.read())

    source_domain = get_mapping_domain(mapping_file_1)
    target_domain = get_mapping_domain(mapping_file_2)

    # Initialize the driver
    with GraphDatabase.driver(
        NEO4J_URI,
        max_connection_pool_size=n_threads,
        auth=(NEO4J_USER, NEO4J_PASS),
        database=NEO4J_DBNAME,
    ) as driver:
        try:
            driver.verify_connectivity()
        except DriverError as e:
            logger.error(e)
            exit(1)

        # create query template given the maximum number of hops
        query = get_query(max_hops, shortest_path, source_domain, target_domain)

        def find_path(first_item: str, second_item: str) -> None:
            """
            It performs a query to find the paths between the given pair of items on Neo4j.

            :param first_item: wikidata id of the first item
            :param second_item: wikidata id of the second item
            """
            with driver.session(database=NEO4J_DBNAME) as session:
                session.execute_write(
                    lambda tx: tx.run(
                        query, first_item=first_item, second_item=second_item
                    )
                )

        # get pairs for which the paths have to be generated
        pairs = product(m_1, m_2)
        n_pairs = len(m_1) * len(m_2)
        # use parallel computing to perform queries
        try:
            ParallelTqdm(n_jobs=n_threads, prefer="threads", total_tasks=n_pairs)(
                delayed(find_path)(first_item, second_item)
                for (first_item, second_item) in pairs
            )
        except KeyboardInterrupt:
            logger.info("Manual interrupt detected. Gracefully quitting")
            driver.close()  # TODO: currently not graceful, needs work
            exit(0)
        except (DriverError, Neo4jError) as e:
            logger.error(e)
            exit(1)


def get_query(
    max_hops: int, shortest_path: bool, source_domain: str, target_domain: str
) -> str:
    """
    This function creates the query for Neo4j based on the given number of hops.

    :param max_hops: max number of hops allowed for the path
    :param shortest_path: whether to configure the query to find just the shortest path or not
    :param source_domain: source domain name
    :param target_domain: target domain name
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
        query = f"""
            MATCH (n1:entity {{wikidata_id: $first_item}})
            MATCH (n2:entity {{wikidata_id: $second_item}})
            
            WITH n1, n2
            OPTIONAL MATCH (n1)-[r:precomputed]-(n2)
            WITH n1, n2, COUNT(r) > 0 AS pathExists
            
            WHERE pathExists = false
            MATCH p=shortestPath((n1)-[r:relation*1..{max_hops}]-(n2))
            WITH n1, n2, length(p) AS pathLength, [n IN nodes(p) | n] AS nodeList, [r IN relationships(p) | r] AS relList
            WITH n1, n2, pathLength, reduce(s = "", i IN range(0, size(relList)-1) |
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
            CREATE (n1)-[r1:precomputed {{path_length: pathLength, path_string: pathString, source_domain: {source_domain}, target_domain: {target_domain}}}]->(n2)
            """
    return query
