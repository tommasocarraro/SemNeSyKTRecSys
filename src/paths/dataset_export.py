from collections import defaultdict
from os import makedirs, remove
from os.path import join, splitext
from typing import Any

import orjson
from loguru import logger
from neo4j import GraphDatabase
from neo4j.exceptions import DriverError, Neo4jError
from tqdm import tqdm

from config import NEO4J_PASS, NEO4J_URI, NEO4J_USER


def dataset_export(
    database_name: str, export_dir_path: str, domain_pairs: list[dict[str, Any]], mappings_file_paths: dict[str, Any]
) -> None:
    """
    Dumps the precomputed shortest paths from the neo4j database for all the given pairs of domains in json format.
    Requires the following settings in apoc.conf:
    apoc.export.file.enabled=true
    apoc.import.file.use_neo4j_config=false

    :param database_name: name of the neo4j database
    :param export_dir_path: directory to export to
    :param domain_pairs: pairs of domains to export
    :param mappings_file_paths: dictionary containing file paths pointing to the domains' mappings
    """
    dump_output_paths = dump_from_neo4j(
        database_name=database_name, export_dir_path=export_dir_path, domain_pairs=domain_pairs
    )

    postprocess_neo4j_dump(dump_output_paths=dump_output_paths, mappings_file_paths=mappings_file_paths)


def dump_from_neo4j(
    database_name: str, export_dir_path: str, domain_pairs: list[dict[str, Any]]
) -> list[dict[str, str]]:
    """
    Dumps the precomputed shortest paths from the neo4j database for all the given pairs of domains in jsonl format.
    Requires the following settings in apoc.conf:
    apoc.export.file.enabled=true
    apoc.import.file.use_neo4j_config=false

    :param database_name: name of the neo4j database
    :param export_dir_path: directory to export to
    :param domain_pairs: pairs of domains to export

    :return: the list of jsonl files to which the database export to
    """
    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS), database=database_name) as driver:
        # make sure that the export dir exists, if not create it
        makedirs(export_dir_path, exist_ok=True)

        # test connectivity to neo4j
        try:
            driver.verify_connectivity()
        except DriverError as e:
            logger.error(e)
            exit(1)

        # query literals used for the creation of indexes
        index_queries: list[str] = [
            "CREATE LOOKUP INDEX rel_label_lookup_index IF NOT EXISTS FOR ()-[r]-()             ON EACH type(r)",
            "CREATE TEXT   INDEX rel_text_index_source  IF NOT EXISTS FOR ()-[r:precomputed]-() ON (r.source_domain)",
            "CREATE TEXT   INDEX rel_text_index_target  IF NOT EXISTS FOR ()-[r:precomputed]-() ON (r.target_domain)",
        ]

        try:
            with driver.session(database=database_name) as session:
                logger.info("Creating the indexes if they don't exist")
                for index_query in index_queries:
                    session.run(index_query)  # type: ignore

                dump_output_paths = []
                for pair_dict in domain_pairs:
                    source_domain = pair_dict["source"]
                    target_domain = pair_dict["target"]
                    logger.info(f"Dumping precomputed paths for {source_domain}->{target_domain}")

                    output_file_path = join(export_dir_path, source_domain + "->" + target_domain + ".jsonl")
                    dump_output_paths.append(
                        {"file_path": output_file_path, "source_domain": source_domain, "target_domain": target_domain}
                    )

                    # the query doesn't actually return anything, I'm consuming the result in order to force the query
                    # to be run eagerly instead of lazily
                    res = session.run(create_query(source_domain, target_domain, output_file_path))  # type: ignore
                    res.consume()
        except (DriverError, Neo4jError) as e:
            logger.error(e)
            exit(1)
        return dump_output_paths


def postprocess_neo4j_dump(dump_output_paths: list[dict[str, str]], mappings_file_paths: dict[str, Any]) -> None:
    """
    Postprocesses the jsonl files into json files for easier handling in the model's pipeline.

    :param dump_output_paths: TODO
    :param mappings_file_paths: dictionary containing file paths pointing to the domains' mappings
    """
    logger.info("Reading all mappings into memory")
    mappings = defaultdict(dict)
    for domain_name, obj in mappings_file_paths.items():
        with open(obj["mapping_file_path"], "rb") as mapping_file:
            mappings[domain_name] = {
                obj["wiki_id"]: asin
                for asin, obj in orjson.loads(mapping_file.read()).items()
                if not isinstance(obj, str)
            }

    logger.info("Postprocessing the dumps")
    for output_dict in tqdm(dump_output_paths, dynamic_ncols=True, desc="Postprocessing the dumps...", leave=False):
        jsonl_file_path = output_dict["file_path"]
        source_domain = output_dict["source_domain"]
        target_domain = output_dict["target_domain"]

        output_file_path = splitext(jsonl_file_path)[0] + ".json"

        with open(jsonl_file_path, "rb") as jsonl_file, open(output_file_path, "wb") as output_file:
            final_dict = defaultdict(lambda: defaultdict(list))

            for line in jsonl_file:
                item = orjson.loads(line)
                path = item["path"]
                path_length = item["path_length"]
                n1_wiki_id = item["n1_wiki_id"]
                n1_asin = mappings[source_domain][n1_wiki_id]
                n2_wiki_id = item["n2_wiki_id"]
                n2_asin = mappings[target_domain][n2_wiki_id]

                final_dict[n1_asin][n2_asin].append({"path_str": path, "path_length": path_length})

            # write the final json file to file system
            output_file.write(orjson.dumps(final_dict, option=orjson.OPT_INDENT_2))
            # delete the jsonl file as it's no longer needed
            remove(jsonl_file_path)


def create_query(source_domain: str, target_domain: str, output_file_path: str) -> str:
    """
    Creates the query used to export the paths from neo4j for a given pair of domains.

    :param source_domain: the name of the source domain
    :param target_domain: the name of the target domain
    :param output_file_path: the path to the output file
    :return: the query used to export the paths from neo4j for a given pair of domains
    """
    return f"""
        CALL apoc.export.json.query(
            "MATCH (n1:entity)-[r:precomputed]->(n2:entity)
             WHERE r.source_domain = {"'" + source_domain + "'"} AND r.target_domain = {"'" + target_domain + "'"} 
             RETURN n1.wikidata_id AS n1_wiki_id, n2.wikidata_id AS n2_wiki_id, r.path_string AS path, r.path_length AS path_length",
            "{output_file_path}",
           {{}}
        )
    """
