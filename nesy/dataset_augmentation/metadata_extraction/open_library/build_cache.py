import os.path
from pathlib import Path

from loguru import logger
from psycopg import connect

from config import PSQL_CONN_STRING
from .utils import _copy_into_table, _reset_database, _load_sql


def build_cache(
    authors_dump_path: str, editions_dump_path: str, works_dump_path: str
) -> None:
    """
    Imports authors and editions metadata from open library into a PostgreSQL database
    Args:
        authors_dump_path: preprocessed authors JSONL file path
        editions_dump_path: preprocessed editions JSONL file path
        works_dump_path: preprocessed works JSONL file path

    Returns:
        None
    """
    sql_base_path = os.path.join(Path(__file__).parent, "SQL")

    # drop open_library database to start clean
    _reset_database()

    conn = connect(PSQL_CONN_STRING)
    conn.autocommit = True
    c = conn.cursor()

    # load the extension for trigram fuzzy matches
    c.execute("CREATE EXTENSION pg_trgm")

    # create the tables
    sql_tables = _load_sql(os.path.join(sql_base_path, "tables.sql"))
    c.execute(sql_tables)

    # import the authors dump
    with c.copy("COPY authors (key, name, name_query) FROM STDIN") as copy:
        _copy_into_table(
            copy,
            authors_dump_path,
            ["key", "name", "name_query"],
            tqdm_desc="Importing authors...",
        )

    # import the editions dump
    with c.copy(
        "COPY editions (key, title, title_query, authors, year, works, isbns) FROM STDIN"
    ) as copy:
        obj_keys = ["key", "title", "title_query", "authors", "year", "works", "isbns"]
        _copy_into_table(
            copy_context=copy,
            input_file_path=editions_dump_path,
            obj_keys=obj_keys,
            tqdm_desc="Importing editions...",
        )

    # import the works dump
    with c.copy("COPY works (key, title, authors) FROM STDIN") as copy:
        _copy_into_table(
            copy_context=copy,
            input_file_path=works_dump_path,
            obj_keys=["key", "title", "authors"],
            tqdm_desc="Importing works...",
        )

    logger.info("Creating the materialized views...")
    sql_views = _load_sql(os.path.join(sql_base_path, "views.sql"))
    c.execute(sql_views)

    logger.info("Creating the indexes...")
    sql_indexes = _load_sql(os.path.join(sql_base_path, "indexes.sql"))
    c.execute(sql_indexes)

    logger.info("Creating the functions...")
    sql_functions = _load_sql(os.path.join(sql_base_path, "functions.sql"))
    c.execute(sql_functions)

    c.execute("ANALYZE")

    # Close the connection
    c.close()
    conn.close()
