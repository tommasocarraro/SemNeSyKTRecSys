import logging
import json
import os
from joblib import delayed
import traceback
from neo4j import GraphDatabase
from nesy.utils import ParallelTqdm
import itertools
from .utils import get_rating_stats, get_cold_start, refine_cold_start_items, get_popular, refine_popular_items


def neo4j_path_finder(mapping_file_1: str, mapping_file_2: str, path_file: str, max_hops: int = 2,
                      shortest_path: bool = True, cold_start: bool = False, popular: bool = False,
                      n_cores: int = 1) -> None:
    """
    This function computes all the available Wikidata's paths between matched entities in mapping_file_1 and
    matched entities in mapping_file_2. It saves all these paths in a JSON file.

    :param mapping_file_1: first mapping file
    :param mapping_file_2: second mapping file
    :param path_file: path where to save the final JSON file containing paths
    :param max_hops: maximum number of hops allowed for the path
    :param shortest_path: whether to find just the shortest path or all the paths connecting the two entities
    :param cold_start: whether to compute paths just for cold-start items in the target domain
    :param popular: whether to compute paths just for popular items in the source domain
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
            pop = get_popular(stats, 30)
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
            cs = get_cold_start(stats, 5)
            m_2 = refine_cold_start_items(cs, mapping_file_2)
    # check if a path file for the given mapping files already exists
    temp_dict = {}
    if os.path.exists(path_file):
        # if it exists, we load a temp dictionary containing the found paths
        with open(path_file) as json_file:
            temp_dict = json.load(json_file)
    # create logger for logging everything to file in case the long executions are interrupted
    # Configure the logger
    logging.basicConfig(level=logging.INFO)  # Set the desired log level
    # Remove all handlers to avoid printing on stdout
    for handler in logging.getLogger().handlers[:]:
        logging.getLogger().removeHandler(handler)
    # Create a FileHandler to write log messages to a file
    file_handler = logging.FileHandler('output.log')
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
        try:
            # check if the paths between these two items have been already computed
            if (first_item not in temp_dict or
                    (first_item in temp_dict and second_item not in temp_dict[first_item])):
                # execute query
                with driver.session() as session:
                    paths = session.execute_read(execute_query, query, first_item, second_item)
                save_paths(first_item, second_item, paths)
        except Exception:
            print(traceback.format_exc())
            logging.info("%s -/- %s -/- exception -/- 0" % (first_item, second_item))

    # computing total number of tasks
    total_tasks = compute_n_tasks(m_1, m_2)
    # get pairs for which the paths have to be generated
    pairs = get_pairs(m_1, m_2)
    # use parallel computing to perform HTTP requests
    try:
        ParallelTqdm(n_jobs=n_cores, prefer="threads", total_tasks=total_tasks)(
            delayed(find_path)(pair) for pair in pairs)
    except (KeyboardInterrupt, Exception):
        print(traceback.format_exc())
        update_file(file_handler, temp_dict, path_file)
        print("Interruption occurred! Path file has been saved!")
        exit()

    update_file(file_handler, temp_dict, path_file)


def update_file(file_handler, temp_dict, path_file):
    """
    This function reads the data that has been logged and creates a JSON file containing this data.

    :param file_handler: file handler of the file where the logging has been written
    :param temp_dict: dictionary containing the paths
    :param path_file: path to the JSON file where to save the paths
    """
    # close the file handler
    file_handler.close()
    # create dictionary with new retrieved data
    with open('./output.log', 'r') as file:
        for line in file:
            split_line = line.split(" -/- ")
            id_1, id_2, msg, length = split_line
            msg = msg.strip()
            length = int(length)
            if id_1 not in temp_dict:
                if msg not in ["no_paths", "exception"]:
                    temp_dict[id_1] = {id_2: [{"path_str": msg, "path_length": length}]}
                else:
                    temp_dict[id_1] = {id_2: msg}
            else:
                if id_2 not in temp_dict[id_1]:
                    if msg not in ["no_paths", "exception"]:
                        temp_dict[id_1][id_2] = [{"path_str": msg, "path_length": length}]
                    else:
                        temp_dict[id_1][id_2] = msg
                else:
                    if isinstance(temp_dict[id_1][id_2], list):
                        temp_dict[id_1][id_2].append({"path_str": msg, "path_length": length})
    # save to file - if the file was already existing, it will be updated. If not, it will be created
    with open(path_file, 'w', encoding='utf-8') as f:
        json.dump(temp_dict, f, ensure_ascii=False, indent=4)
    # delete temporary log file
    os.remove("./output.log")


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
        query = ("MATCH path=shortestPath(%s-[*1..%d]-%s) RETURN path, "
                 "length(path) AS path_length") % (query_head, max_hops, query_tail)
    return query


def execute_query(tx, query, first_item, second_item):
    """
    This function executes the given query on Neo4j.
    :param tx:
    :param query: the query to be executed
    :param first_item: first query parameter value
    :param second_item: second query parameter value
    :return: the results of the query
    """
    result = tx.run(query, first_item=first_item, second_item=second_item)
    return [(record["path"], record["path_length"]) for record in result]


def save_paths(first_item: str, second_item: str, paths: list) -> None:
    """
    This function logs the found paths.

    :param first_item: first item
    :param second_item: second item
    :param paths: list of found paths
    """
    if paths:
        for path in paths:
            nodes = list(path[0].nodes)
            relationships = list(path[0].relationships)

            path_str = ""
            for i in range(len(nodes)):
                node = nodes[i]
                path_str += f" {node['label']} ({node['wikidata_id']})"

                if i < len(relationships):
                    relationship = relationships[i]
                    # Assume the relationship has a 'label' attribute you want to display
                    direction = "-->" if relationship.start_node.element_id == nodes[i].element_id else "<--"
                    path_str += f" {direction} {relationship['label']} ({relationship['wikidata_id']}) {direction}"

            path_str = path_str.strip()
            # log the path
            logging.info("%s -/- %s -/- %s -/- %d" % (first_item, second_item, path_str, path[1]))
    else:
        logging.info("%s -/- %s -/- no_paths -/- 0" % (first_item, second_item))


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


def get_pairs(source_d: dict, target_d: dict) -> tuple:
    """
    This function computes the pairs for which the paths have to be generated and return a generator of pairs.

    :param source_d: source domain dict
    :param target_d: target domain dict
    :return: pairs for which the paths have to be generated returned as a generator
    """
    if not isinstance(source_d, list):
        source_d_ids = [data["wiki_id"] for asin, data in source_d.items() if isinstance(source_d[asin], dict)]
    else:
        source_d_ids = source_d
    if not isinstance(target_d, list):
        target_d_ids = [data["wiki_id"] for asin, data in target_d.items() if isinstance(target_d[asin], dict)]
    else:
        target_d_ids = target_d
    return itertools.product(source_d_ids, target_d_ids)
