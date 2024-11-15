import json
from collections import defaultdict
from os import makedirs
from os.path import abspath, join
from typing import LiteralString, OrderedDict

import orjson
from loguru import logger
from neo4j import GraphDatabase
from neo4j.exceptions import DriverError, Neo4jError
from tqdm import tqdm

from config import NEO4J_PASS, NEO4J_URI, NEO4J_USER


def dataset_export(
    database_name: str, export_dir_path: str, domains: list[tuple[str, str]]
) -> None:
    """
    Requires in apoc.conf:
    apoc.export.file.enabled=true
    apoc.import.file.use_neo4j_config=false
    Args:
        database_name:
        export_dir_path:
        domains:

    Returns:

    """
    with GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS), database=database_name
    ) as driver:
        makedirs(export_dir_path, exist_ok=True)

        try:
            driver.verify_connectivity()
        except DriverError as e:
            logger.error(e)
            exit(1)

        index_queries: list[LiteralString] = [
            "CREATE LOOKUP INDEX rel_label_lookup_index IF NOT EXISTS FOR ()-[r]-()             ON EACH type(r)",
            "CREATE TEXT   INDEX rel_text_index_source  IF NOT EXISTS FOR ()-[r:precomputed]-() ON (r.source_domain)",
            "CREATE TEXT   INDEX rel_text_index_target  IF NOT EXISTS FOR ()-[r:precomputed]-() ON (r.target_domain)",
        ]

        try:
            with driver.session(database=database_name) as session:
                logger.info("Creating the indexes if they don't exist")
                for index_query in index_queries:
                    session.run(index_query)

                file_paths = []
                for source_domain, target_domain in domains:
                    logger.info(
                        f"Dumping precomputed paths for {source_domain}->{target_domain}"
                    )
                    base_output_file_path = join(
                        export_dir_path,
                        source_domain + "->" + target_domain,
                    )

                    temp_file_path = base_output_file_path + ".jsonl"
                    output_file_path = base_output_file_path + ".json"
                    file_paths.append((temp_file_path, output_file_path))

                    res = session.run(
                        create_query(source_domain, target_domain, abspath(temp_file_path))  # type: ignore
                    )
                    res.consume()

                logger.info("Postprocessing the dumps")
                for temp_file_path, output_file_path in tqdm(
                    file_paths, dynamic_ncols=True, desc="Postprocessing the dumps..."
                ):
                    with open(temp_file_path, "r") as temp_file, open(
                        output_file_path, "wb"
                    ) as output_file:
                        final_dict = {}

                        for line in temp_file:
                            item = json.loads(line)
                            n1_asin = item["n1_asin"]
                            n2_asin = item["n2_asin"]
                            path = item["path"]
                            path_length = item["path_length"]

                            if n1_asin not in final_dict:
                                final_dict[n1_asin] = {}

                            if n2_asin not in final_dict[n1_asin]:
                                final_dict[n1_asin][n2_asin] = []

                            final_dict[n1_asin][n2_asin].append(
                                {"path_str": path, "path_length": path_length}
                            )

                        output_file.write(
                            orjson.dumps(
                                dict(sorted(final_dict.items())),
                                option=orjson.OPT_INDENT_2,
                            )
                        )

        except (DriverError, Neo4jError) as e:
            logger.error(e)
            exit(1)


def create_query(source_domain: str, target_domain: str, output_file_path: str) -> str:
    return f"""
        CALL apoc.export.json.query(
            "MATCH (n1:entity)-[r:precomputed]-(n2:entity)
             WHERE r.source_domain = {"'" + source_domain + "'"} AND r.target_domain = {"'" + target_domain + "'"} 
             RETURN n1.amazon_asin AS n1_asin, n2.amazon_asin AS n2_asin, r.path_string AS path, r.path_length AS path_length",
            "{output_file_path}",
           {{}}
        )
    """
