import orjson as json
from loguru import logger
from psycopg import connect
from tqdm.auto import tqdm

from config import PSQL_CONN_STRING, PSQL_CONN_STRING_SANS_DB


def build_cache(authors_dump_path: str, editions_dump_path: str) -> None:
    """
    Imports authors and editions metadata from open library into a PostgreSQL database
    Args:
        authors_dump_path: preprocessed authors JSONL file path
        editions_dump_path: preprocessed editions JSONL file path

    Returns:
        None
    """
    # drop open_library database to start clean
    conn = connect(PSQL_CONN_STRING_SANS_DB)
    conn.autocommit = True  # drop statements will raise exceptions without autocommit
    c = conn.cursor()
    c.execute("DROP DATABASE IF EXISTS open_library WITH (FORCE)")
    c.execute("CREATE DATABASE open_library OWNER postgres")
    c.close()
    conn.close()
    # psycopg doesn't allow selecting a database without reconnecting
    conn = connect(PSQL_CONN_STRING)
    conn.autocommit = False
    c = conn.cursor()

    # c.execute("CREATE EXTENSION fuzzystrmatch")
    c.execute("CREATE EXTENSION pg_trgm")
    c.execute("CREATE TABLE authors (key TEXT PRIMARY KEY, name TEXT)")
    c.execute(
        "CREATE TABLE editions (key TEXT PRIMARY KEY, title VARCHAR(255), authors TEXT[], year CHAR(4))"
    )

    # Process the authors dump
    with c.copy("COPY authors (key, name) FROM STDIN") as copy:
        with open(authors_dump_path, "r", encoding="utf-8") as author_file:
            for line in tqdm(
                author_file, desc="Importing authors...", dynamic_ncols=True
            ):
                author = json.loads(line)
                copy.write_row((author["key"], author["name"]))
    conn.commit()

    # Process the editions dump
    with c.copy("COPY editions (key, title, authors, year) FROM STDIN") as copy:
        with open(editions_dump_path, "r", encoding="utf-8") as edition_file:
            for line in tqdm(
                edition_file, desc="Importing editions...", dynamic_ncols=True
            ):
                edition = json.loads(line)
                copy.write_row(
                    (
                        edition["key"],
                        edition["title"],
                        edition["authors"],
                        edition["year"],
                    )
                )
    conn.commit()

    logger.info("Creating the materialized view...")
    c.execute(
        """
        CREATE MATERIALIZED VIEW edition_authors_materialized_view AS
        SELECT e.title, e.year, array_agg(a.name) AS authors
        FROM editions e
        JOIN authors a ON a.key = ANY (e.authors)
        GROUP BY e.title, e.year"""
    )
    logger.info("Creating the materialized view index...")
    c.execute(
        "CREATE INDEX trgm_idx ON edition_authors_materialized_view USING gist (title gist_trgm_ops)"
    )
    conn.commit()

    # Close the connection
    c.close()
    conn.close()
