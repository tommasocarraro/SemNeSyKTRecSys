import threading
from itertools import product, takewhile
from typing import Optional

import orjson
from joblib import delayed
from loguru import logger
from neo4j import GraphDatabase
from neo4j.exceptions import DriverError, Neo4jError

from config import NEO4J_PASS, NEO4J_URI, NEO4J_USER
from .FilePaths import FilePaths
from .neo4j_make_query import make_query
from .utils import (
    ParallelTqdm,
    get_cold_start_items,
    get_popular_items,
    get_rating_stats,
    refine_cold_start_items,
    refine_popular_items,
)


def neo4j_path_finder(
    database_name: str,
    file_paths: FilePaths,
    max_hops: int = 2,
    shortest_path: bool = True,
    cs_threshold: Optional[int] = 5,
    pop_threshold: Optional[int] = 50,
    n_threads: int = -1,
) -> None:
    """
    This function computes all the available Wikidata's paths between matched entities in mapping_file_1 and
    matched entities in mapping_file_2. It saves all these paths in a JSON file.

    :param database_name: name of the neo4j database
    :param file_paths: data class which contains the file paths needed for path finding
    :param max_hops: maximum number of hops allowed for the path
    :param shortest_path: whether to find just the shortest path or all the paths connecting the two entities
    :param cs_threshold: threshold to select cold-start items,
    :param pop_threshold: threshold to select popular items
    the pop_threshold last used and all the items from the source domain of >= pop will be excluded
    :param n_threads: number of processors to be used to execute this function
    """

    logger.debug("Loading mappings")
    source_mapping, target_mapping = load_mappings(
        file_paths=file_paths,
        cs_threshold=cs_threshold,
        pop_threshold=pop_threshold,
    )

    with GraphDatabase.driver(
        uri=NEO4J_URI,
        max_connection_pool_size=n_threads,
        auth=(NEO4J_USER, NEO4J_PASS),
        database=database_name,
    ) as driver:
        try:
            logger.debug("Trying to connect to Neo4j")
            driver.verify_connectivity()
            logger.debug("Connected to Neo4j")
        except DriverError as e:
            logger.error(e)
            exit(1)

        logger.debug("Retrieving source-item pairs")
        # get pairs for which the paths have to be generated
        pairs = product(source_mapping, target_mapping)

        # declare a threading event to transmit keyboard interrupts to the generator
        stop_flag = threading.Event()

        # create a generator which stops when keyboard interrupts are detected
        pairs_generator = (
            item for item in takewhile(lambda _: not stop_flag.is_set(), pairs)
        )

        # create query template given the maximum number of hops
        query = make_query(
            max_hops=max_hops,
            shortest_path=shortest_path,
            source_domain=file_paths.source_domain_name,
            target_domain=file_paths.target_domain_name,
        )

        def find_path(first_item: str, second_item: str) -> None:
            """
            It performs a query to find the paths between the given pair of items on Neo4j.

            :param first_item: wikidata id of the first item
            :param second_item: wikidata id of the second item
            """
            # double-checking the types as python does not enforce typing at runtime
            # and neo4j fails the query quietly if the parameters are incorrect
            if not isinstance(first_item, str) or not isinstance(second_item, str):
                logger.error(f"Function find_path only accepts strings as parameters")
                exit(1)
            logger.debug(f"Finding a path from {first_item} to {second_item}")
            with driver.session(database=database_name) as session:
                session.execute_write(
                    lambda tx: tx.run(
                        query, first_item=first_item, second_item=second_item
                    )
                )

        try:
            ParallelTqdm(
                n_jobs=n_threads,
                prefer="threads",
                total_tasks=len(source_mapping) * len(target_mapping),
                desc=f"Computing paths from {file_paths.source_domain_name} to {file_paths.target_domain_name}",
            )(
                delayed(find_path)(first_item, second_item)
                for (first_item, second_item) in pairs_generator
            )
        except KeyboardInterrupt:
            logger.info("Manual interrupt detected. Gracefully quitting")
            stop_flag.set()
        except (DriverError, Neo4jError) as e:
            logger.error(e)
            exit(1)
        finally:
            driver.close()
            if stop_flag.is_set():
                exit(0)


def load_mappings(
    file_paths: FilePaths,
    pop_threshold: Optional[int] = None,
    cs_threshold: Optional[int] = None,
) -> tuple[list[str], list[str]]:
    """
    Loads source and target mappings. If both pop_threshold and cs_threshold are set, restrict the items based on said
    thresholds through the ratings files.

    :param file_paths: data class which contains the file paths needed for path finding
    :param pop_threshold: threshold to select popular items
    :param cs_threshold: threshold to select cold-start items
    :return: source and target mappings
    """
    # load both domains' full mappings from the json files
    with open(file_paths.mapping_source_domain, "rb") as mapping_source_file:
        source_mapping_json = orjson.loads(mapping_source_file.read())
    with open(file_paths.mapping_target_domain, "rb") as mapping_target_file:
        target_mapping_json = orjson.loads(mapping_target_file.read())

    if pop_threshold is not None and cs_threshold is not None:
        # if both pop and cs thresholds are set, filter the mappings so they only include the items within the parameters
        source_stats = get_rating_stats(file_paths.reviews_source_domain, "item")

        # retrieve the list of mappings to compute
        pop_list = get_popular_items(source_stats, pop_threshold)
        source_mapping = refine_popular_items(pop_list, source_mapping_json)

        # get the filtered mapping for the target domain
        target_stats = get_rating_stats(file_paths.reviews_target_domain, "item")
        cs_list = get_cold_start_items(target_stats, cs_threshold)
        target_mapping = refine_cold_start_items(cs_list, target_mapping_json)

    elif pop_threshold is None and cs_threshold is None:
        # if the thresholds aren't set simply convert the dicts to lists of wikidata ids
        source_mapping = [
            obj["wiki_id"]
            for obj in source_mapping_json.values()
            if isinstance(obj, dict)
        ]
        target_mapping = [
            obj["wiki_id"]
            for obj in target_mapping_json.values()
            if isinstance(obj, dict)
        ]

    else:
        logger.error(
            "Params cs_threshold and pop_threshold must be either both set or both None"
        )
        exit(1)

    return source_mapping, target_mapping
