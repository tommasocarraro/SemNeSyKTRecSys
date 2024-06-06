import json
import os
import sqlite3

from loguru import logger
from tqdm import tqdm

from nesy.dataset_augmentation.metadata_extraction.utils import (
    object_recursive_delete_fields,
)

FIELDS_TO_BE_REMOVED = [
    "subtitle",
    "average_rating",
    "rating_number",
    "description",
    "price",
    "images",
    "videos",
    "store",
    "Audio language",
    "Subtitles",
    "bought_together",
    "author.about",
    "author.avatar",
    "details.Discontinued By Manufacturer",
    "details.Is Discontinued By Manufacturer",
    "details.Dimensions",
    "details.Product Dimensions",
    "details.Package Dimensions",
    "details.Format",
    "details.Media Format",
    "details.Date First Available",
    "details.Paperback",
    "details.Item Weight",
    "details.Manufacturer",
    "details.Run time",
    "details.Number of discs",
    "details.Reading age",
    "details.Grade level",
    "details.Calendar",
    "features",
    "main_category",
]


def _insert_file_into_database(
    file_path: str, cursor: sqlite3.Cursor, table_name: str
) -> None:
    # read the JSONL file and insert data into the SQLite table
    with open(file_path, "r") as file:
        for line in tqdm(file, desc=f"Importing JSONL data from {file_path}"):
            try:
                json_obj = json.loads(line.strip())

                # filter unneeded information
                object_recursive_delete_fields(json_obj, FIELDS_TO_BE_REMOVED)

                #
                if "title" in json_obj and json_obj["title"] is not None:
                    parent_asin = json_obj.get("parent_asin")
                    data = json.dumps(json_obj, indent=4, ensure_ascii=False)
                    cursor.execute(
                        f"INSERT INTO {table_name} (parent_asin, data) VALUES (?, ?)",
                        (parent_asin, data),
                    )
            except json.JSONDecodeError as e:
                logger.error(f"There was en error while decoding the JSON object: {e}")
                continue


def build_sqlite_cache(file_paths: list[str], cache_path: str) -> None:
    """
    Build a sqlite cache containing the data from each file in file_paths, using the file_name without
    extension as table name.
    Args:
        file_paths: list of file paths.
        cache_path: path to save the sqlite cache to.
    """
    # delete old cache if it already exists
    if os.path.exists(cache_path):
        os.remove(cache_path)

    # connect to the database, if file does not exist it will be created
    conn = sqlite3.connect(cache_path)
    cursor = conn.cursor()

    for file_path in file_paths:
        # derive the table name from the file name
        table_name = os.path.splitext(os.path.basename(file_path))[0]

        # create a table with the necessary fields
        cursor.execute(
            f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY,
            parent_asin TEXT,
            data TEXT
        )
        """
        )
        conn.commit()

        # insert current file into the database
        _insert_file_into_database(file_path, cursor, table_name)

        # create an index on the parent_asin field for faster queries
        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS idx_parent_asin_{table_name} ON {table_name} (parent_asin)"
        )

    # commit the changes and close the connection
    conn.commit()
    conn.close()

    logger.info("Data import complete.")
