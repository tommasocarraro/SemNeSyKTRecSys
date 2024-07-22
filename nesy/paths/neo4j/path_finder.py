import logging
import json
import os
from joblib import delayed
import traceback
from neo4j import GraphDatabase
from nesy.utils import ParallelTqdm


def neo4j_path_finder(mapping_file_1: str, mapping_file_2: str, path_file: str, max_hops: int = 2,
                      n_cores: int = 1) -> None:
    """
    This function computes all the available Wikidata's paths between matched entities in mapping_file_1 and
    matched entities in mapping_file_2. It saves all these paths in a JSON file.

    :param mapping_file_1: first mapping file
    :param mapping_file_2: second mapping file
    :param path_file: path where to save the final JSON file containing paths
    :param max_hops: maximum number of hops allowed for the path
    :param n_cores: number of processors to be used to execute this function
    """
    # read mapping files
    with open(mapping_file_1) as json_file:
        m_1 = json.load(json_file)
    with open(mapping_file_2) as json_file:
        m_2 = json.load(json_file)
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

    def find_path(first_item: str, second_item: str) -> None:
        """
        It performs a query to find the paths between the given items on Neo4j.

        :param first_item: head of the path
        :param second_item: tail of the path
        """
        try:
            # check if the paths between these two items have been already computed
            if (first_item not in temp_dict or
                    (first_item in temp_dict and second_item not in temp_dict[first_item])):
                # define query to launch
                query = get_query(max_hops)
                # execute query
                with driver.session() as session:
                    paths = session.execute_read(execute_query, query, first_item, second_item)
                save_paths(first_item, second_item, paths, temp_dict)
        except Exception:
            print(traceback.format_exc())
            logging.info("%s -/- %s -/- exception" % (first_item, second_item))

    # computing total number of tasks
    total_tasks = compute_n_tasks(m_1, m_2)
    # use parallel computing to perform HTTP requests
    try:
        ParallelTqdm(n_jobs=n_cores, prefer="threads", total_tasks=total_tasks)(
            delayed(find_path)(m_1[first_item]["wiki_id"],
                               m_2[second_item]["wiki_id"])
            for first_item in m_1
            for second_item in m_2
            if isinstance(m_1[first_item], dict)
            if isinstance(m_2[second_item], dict))
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
            id_1, id_2, msg = split_line
            msg = msg.strip()
            if id_1 not in temp_dict:
                if msg not in ["no_paths", "exception"]:
                    temp_dict[id_1] = {id_2: [msg]}
                else:
                    temp_dict[id_1] = {id_2: msg}
            else:
                if id_2 not in temp_dict[id_1]:
                    if msg not in ["no_paths", "exception"]:
                        temp_dict[id_1][id_2] = [msg]
                    else:
                        temp_dict[id_1][id_2] = msg
                else:
                    if isinstance(temp_dict[id_1][id_2], list):
                        temp_dict[id_1][id_2].append(msg)
    # save to file - if the file was already existing, it will be updated. If not, it will be created
    with open(path_file, 'w', encoding='utf-8') as f:
        json.dump(temp_dict, f, ensure_ascii=False, indent=4)
    # delete temporary log file
    os.remove("./output.log")


def get_query(max_hops: int) -> str:
    """
    This function creates the query for Neo4j based on the given number of hops.

    :param max_hops: max number of hops allowed for the path
    :return: the query to be executed
    """
    query = ""
    query_head = "(n1:entity {wikidata_id: $first_item})"
    query_tail = "(n2:entity {wikidata_id: $second_item})"
    for i in range(max_hops):
        query += "MATCH p%d=%s" % (i + 1, query_head)
        for j in range(i + 1):
            query += "-[*1..1]-(mid%d:entity)" % (j + 1,)
        query += "-[*1..1]-"
        query += "%s RETURN p%d AS path" % (query_tail, i + 1)
        if i != max_hops - 1:
            query += " UNION "
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
    return [record["path"] for record in result]


def save_paths(first_item: str, second_item: str, paths: list, temp_dict: dict) -> None:
    """
    This function logs the found paths.

    :param first_item: first item
    :param second_item: second item
    :param paths: list of found paths
    :param temp_dict: dictionary that has to be updated with new results
    """
    if paths:
        for path in paths:
            nodes = list(path.nodes)
            relationships = list(path.relationships)

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
            logging.info("%s -/- %s -/- %s" % (first_item, second_item, path_str))
    else:
        logging.info("%s -/- %s -/- no_paths" % (first_item, second_item))


def compute_n_tasks(mapping_1: dict, mapping_2: dict) -> int:
    """
    This function computes the total number of pairs for which the paths have to be computed.

    :param mapping_1: dictionary containing the mapping between Amazon and wikidata in the source domain
    :param mapping_2: dictionary containing the mapping between Amazon and wikidata in the target domain
    :return: number of pairs for which the paths have to be computed
    """
    matched_items_1, matched_items_2 = 0, 0
    for item in mapping_1:
        if isinstance(mapping_1[item], dict):
            matched_items_1 += 1
    for item in mapping_2:
        if isinstance(mapping_2[item], dict):
            matched_items_2 += 1
    return matched_items_1 * matched_items_2
