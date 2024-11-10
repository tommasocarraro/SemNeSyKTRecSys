import os.path
from typing import LiteralString

from loguru import logger
from neo4j import GraphDatabase
from neo4j.exceptions import DriverError, Neo4jError

from config import NEO4J_DBNAME, NEO4J_PASS, NEO4J_URI, NEO4J_USER


def export_paths(export_dir_path: str, domains: list[tuple[str, str]]) -> None:
    with GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS), database=NEO4J_DBNAME
    ) as driver:
        os.makedirs(export_dir_path, exist_ok=True)

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
            with driver.session(database=NEO4J_DBNAME) as session:
                logger.info("Creating the indexes if they don't exist")
                for index_query in index_queries:
                    session.run(index_query)
                for source_domain, target_domain in domains:
                    logger.info(
                        f"Dumping precomputed paths for {source_domain}->{target_domain}"
                    )
                    file_path = os.path.abspath(
                        os.path.join(
                            export_dir_path,
                            source_domain + "->" + target_domain + ".jsonl",
                        )
                    )
                    session.run(
                        create_query(source_domain, target_domain, file_path),  # type: ignore
                        source_domain=source_domain,
                        target_domain=target_domain,
                        file_path=file_path,
                    )
        except (DriverError, Neo4jError) as e:
            logger.error(e)
            exit(1)


def create_query(source_domain: str, target_domain: str, output_file_path: str) -> str:
    return f"""
        CALL apoc.export.json.query(
            "MATCH (n1:entity)-[r:precomputed]-(n2:entity)
             WHERE r.source_domain = {"'" + source_domain + "'"} AND r.target_domain = {"'" + target_domain + "'"} 
             RETURN n1, r, n2",
            "{output_file_path}",
           {{}}
        )
    """
