import os.path
import re

import orjson as json
import psycopg
from tqdm.auto import tqdm


def clean_input_files():
    with open(
        os.path.join("data", "open_library", "ol_dump_editions_2024-04-30.txt"),
        "r",
        encoding="utf-8",
    ) as in_file:
        with open(
            os.path.join("data", "open_library", "editions.jsonl"), "wb"
        ) as out_file:
            for line in in_file:
                parts = line.split("\t")
                edition = json.loads(parts[4])
                title = edition.get("title", "")
                if len(title) == 0 or len(title) > 200:
                    continue
                publish_date = edition.get("publish_date", "")
                s = re.search(r"((?:19|20)\d{2})", publish_date)
                if s:
                    year = s.group(1)
                    if int(year) > 2014:
                        continue
                else:
                    year = ""
                authors = edition.get("authors", [])
                if len(authors) > 0:
                    if isinstance(authors[0], dict):
                        authors = [
                            auth_dict["key"].split("/")[2] for auth_dict in authors
                        ]
                    else:
                        authors = [auth.split("/")[2] for auth in authors]
                if len(authors) == 0 and year == "":
                    continue
                edition_clean = {
                    "key": edition["key"].split("/")[2],
                    "title": edition["title"],
                    "year": year,
                    "authors": authors,
                }
                out_file.write(json.dumps(edition_clean) + b"\n")
    with open(
        os.path.join("data", "open_library", "ol_dump_authors_2024-04-30.txt"),
        "r",
        encoding="utf-8",
    ) as in_file:
        with open(
            os.path.join("data", "open_library", "authors.jsonl"), "wb"
        ) as out_file:
            for line in in_file:
                parts = line.split("\t")
                author = json.loads(parts[4])
                if "name" not in author:
                    continue
                author_clean = {
                    "key": author["key"].split("/")[2],
                    "name": author["name"],
                }
                out_file.write(json.dumps(author_clean) + b"\n")


def build_cache():
    conn = psycopg.connect(
        user="postgres",
        password="postgres",
        host="localhost",
        port="5432",
    )
    conn.autocommit = True
    c = conn.cursor()
    c.execute("DROP DATABASE IF EXISTS open_library WITH (FORCE)")
    c.execute("CREATE DATABASE open_library OWNER postgres")
    c.close()
    conn.close()
    conn = psycopg.connect(
        dbname="open_library",
        user="postgres",
        password="postgres",
        host="localhost",
        port="5432",
    )
    conn.autocommit = False
    c = conn.cursor()

    c.execute("CREATE EXTENSION fuzzystrmatch")
    c.execute("CREATE EXTENSION pg_trgm")
    c.execute("CREATE TABLE authors (key TEXT PRIMARY KEY, name TEXT)")
    c.execute(
        "CREATE TABLE editions (key TEXT PRIMARY KEY, title VARCHAR(255), authors TEXT[], year CHAR(4))"
    )

    # Process the author dump
    authors_dump_path = os.path.join("data", "open_library", "authors.jsonl")
    with c.copy("COPY authors (key, name) FROM STDIN") as copy:
        with open(authors_dump_path, "r", encoding="utf-8") as author_file:
            for i, line in enumerate(
                tqdm(author_file, desc="Importing authors...", dynamic_ncols=True)
            ):
                author = json.loads(line)
                copy.write_row((author["key"], author["name"]))
    conn.commit()

    # Process the edition dump
    editions_dump_path = os.path.join("data", "open_library", "editions.jsonl")
    with c.copy("COPY editions (key, title, authors, year) FROM STDIN") as copy:
        with open(editions_dump_path, "r", encoding="utf-8") as edition_file:
            for i, line in enumerate(
                tqdm(edition_file, desc="Importing editions...", dynamic_ncols=True)
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

    print("Creating the materialized view...")
    c.execute(
        """
        CREATE MATERIALIZED VIEW edition_authors_materialized_view AS
        SELECT e.title, e.year, array_agg(a.name) AS authors
        FROM editions e
        JOIN authors a ON a.key = ANY (e.authors)
        GROUP BY e.title, e.year"""
    )
    print("Creating the materialized view index...")
    c.execute(
        "CREATE INDEX trgm_idx ON edition_authors_materialized_view USING gist (title gist_trgm_ops)"
    )
    conn.commit()

    # Close the connection
    c.close()
    conn.close()


def fuzzy_search_title(search_term):
    conn = psycopg.connect(
        dbname="open_library",
        user="postgres",
        password="postgres",
        host="localhost",
        port="5432",
    )
    c = conn.cursor()

    query = """
    SELECT title, authors, year
    FROM edition_authors_materialized_view
    WHERE title %% %s
    ORDER BY (title <-> %s)
    LIMIT 5;"""
    c.execute(query, (search_term, search_term))
    results = c.fetchall()
    print(results)

    conn.close()
    return results


def set_fuzzy_threshold(thresh: float) -> None:
    conn = psycopg.connect(
        dbname="open_library",
        user="postgres",
        password="postgres",
        host="localhost",
        port="5432",
    )
    c = conn.cursor()
    c.execute("SET pg_trgm.similarity_threshold = %s", (thresh,))
    conn.commit()
    c.close()
    conn.close()
