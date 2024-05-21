import json
import sqlite3
from typing import Any


def query_by_asin(table_name: str, asin_value: str, cache_path: str) -> list[Any]:
    """
    Given an SQLITE database and a table name, query the database to find the entry with the given asin.
    Args:
        table_name: The name of the table to query.
        asin_value: The asin value to query.
        cache_path: The path to the database to query.

    Returns: A list of entries found in the database.
    """
    conn = sqlite3.connect(cache_path)
    cursor = conn.cursor()
    cursor.execute(
        f"SELECT data FROM {table_name} WHERE parent_asin = ?", (asin_value,)
    )
    result = cursor.fetchall()
    conn.close()
    return [json.loads(row[0]) for row in result]
