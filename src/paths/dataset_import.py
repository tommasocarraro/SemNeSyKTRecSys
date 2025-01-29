import os.path
from os import path

from loguru import logger
from neo4j import GraphDatabase
from neo4j.exceptions import ClientError, DriverError

from config import NEO4J_PASS, NEO4J_URI, NEO4J_USER
from src.paths.utils import run_shell_command


def dataset_import(
    database_name: str, nodes_file_path: str, rels_file_path: str, use_sudo=False
):
    """
    Requires being able to import from any folder, otherwise the file paths need to be relative to the neo4j import folder
    Args:
        database_name:
        nodes_file_path:
        rels_file_path:
        use_sudo:

    Returns:

    """
    if not path.exists(nodes_file_path):
        logger.error(f"Nodes file not found: {nodes_file_path}")
        exit(1)
    if not path.exists(rels_file_path):
        logger.error(f"Relationships file not found: {rels_file_path}")
        exit(1)

    with GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASS),
        database=database_name,
    ) as driver:
        # first verify if the neo4j server is up and running
        try:
            driver.verify_connectivity()
            logger.info("Neo4j up and running")
        except DriverError as e:
            logger.error(e)
            exit(1)

        # then stop the database if it's running
        try:
            with driver.session() as session:
                session.run(f"DROP DATABASE {database_name}")  # type: ignore
            logger.info(f"{database_name} database stopped")
        except (DriverError, ClientError) as e:
            logger.error(e)
            exit(1)

        args = [
            "neo4j-admin",
            "database",
            "import",
            "full",
            database_name,
            f"--nodes={nodes_file_path}",
            f"--relationships={rels_file_path}",
            "--overwrite-destination",
        ]
        run_shell_command(args=args, use_sudo=use_sudo)

        if os.path.exists("import.report"):
            os.remove("import.report")

        try:
            with driver.session() as session:
                session.run(f"CREATE DATABASE {database_name} IF NOT EXISTS")  # type: ignore
            logger.info("Database correctly imported")
        except (DriverError, ClientError) as e:
            logger.error(e)
            logger.error(
                f"Something unexpected occurred. You may need to run the following query to finalize the import:\nCREATE DATABASE {database_name}"
            )
            exit(1)

        try:
            with driver.session(database=database_name) as session:
                session.run(
                    "CREATE TEXT INDEX node_wikidata_id_text_index FOR (n:entity) ON (n.wikidata_id)"
                )
            logger.info("Index created successfully")
        except (DriverError, ClientError) as e:
            logger.error(e)
            logger.error("Failed to create index")
            exit(1)
