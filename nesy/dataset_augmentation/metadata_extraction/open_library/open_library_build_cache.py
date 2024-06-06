import orjson as json
from loguru import logger
from psycopg import connect
from tqdm.auto import tqdm

from config import PSQL_CONN_STRING, PSQL_CONN_STRING_SANS_DB


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

    c.execute("CREATE EXTENSION pg_trgm")
    c.execute("CREATE TABLE authors (key TEXT PRIMARY KEY, name TEXT NOT NULL)")
    c.execute(
        """
        CREATE TABLE editions (
            key TEXT PRIMARY KEY, 
            title VARCHAR(200) NOT NULL, 
            authors TEXT[] NOT NULL, 
            year CHAR(4) NOT NULL,
            works TEXT[] NOT NULL
        )
        """
    )
    c.execute(
        "CREATE TABLE works (key TEXT PRIMARY KEY, title VARCHAR(200) NOT NULL, authors TEXT[] NOT NULL)"
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
    with c.copy("COPY editions (key, title, authors, year, works) FROM STDIN") as copy:
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
                        edition["works"],
                    )
                )
    conn.commit()

    # Process the works dump
    with c.copy("COPY works (key, title, authors) FROM STDIN") as copy:
        with open(works_dump_path, "r", encoding="utf-8") as work_file:
            for line in tqdm(work_file, desc="Importing works...", dynamic_ncols=True):
                work = json.loads(line)
                copy.write_row(
                    (
                        work["key"],
                        work["title"],
                        work["authors"],
                    )
                )

    logger.info("Creating the materialized view for faster querying...")
    c.execute(
        """
        CREATE MATERIALIZED VIEW combined_materialized_view AS
        WITH unnested_authors AS (
            SELECT
                w.key,
                w.title,
                ea.key as author_key
            FROM works w, UNNEST(w.authors) as ea(key)
        )
        SELECT DISTINCT
            ua.key as key,
            ua.title,
            a.name as author_name,
            e.year
        FROM unnested_authors ua
        JOIN authors a ON a.key = ua.author_key
        JOIN editions e ON ua.key = ANY(e.works)
        """
    )
    logger.info("Creating index on title for materialized view...")
    c.execute(
        "CREATE INDEX ON combined_materialized_view USING gin (title gin_trgm_ops)"
    )
    logger.info("Creating index on author names for materialized view...")
    c.execute(
        "CREATE INDEX ON combined_materialized_view USING gin (author_name gin_trgm_ops)"
    )
    logger.info("Creating index on year for previous materialized view...")
    c.execute("CREATE INDEX ON combined_materialized_view (year)")
    c.execute("ANALYZE")

    conn.commit()

    # Close the connection
    c.close()
    conn.close()
