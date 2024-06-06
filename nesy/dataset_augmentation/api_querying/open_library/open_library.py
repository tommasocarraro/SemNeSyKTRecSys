from typing import Optional

from loguru import logger
from psycopg import cursor

from psycopg_pool import ConnectionPool

from config import PSQL_CONN_STRING
from nesy.dataset_augmentation import state
from .queries import make_query

psql_pool: Optional[ConnectionPool] = None


def get_books_info(query_data: list[tuple[str, Optional[str], Optional[str]]]):
    global psql_pool
    if psql_pool is None:
        psql_pool = ConnectionPool(PSQL_CONN_STRING)
    psql_pool.wait()
    if not state.SET_SIM_THRESHOLD:
        set_sim_threshold(0.9, psql_pool)
    data = fuzzy_search_titles(query_data, psql_pool)
    return data


def _execute_queries(cur: cursor, params_list: list[dict[str, str]]):
    queries = []
    for params_dict in params_list:
        queries.append(make_query(params_dict))
    query_results = []
    for query in queries:
        cur.execute(query)
        query_results.append(cur.fetchall())
    return query_results


def fuzzy_search_titles(
    query_data: list[tuple[str, Optional[str], Optional[str]]], pool: ConnectionPool
) -> list[list[tuple]]:
    """
    Performs an asynchronous fuzzy search against the PostgreSQL database through the provided connection pool
    Args:
        query_data: query data to search for
        pool: The connection pool to use

    Returns:
        TODO
    """
    logger.info(f"Performing {len(query_data)} fuzzy searches against Open Library")
    params_list = []
    for i, (title, person, year) in enumerate(query_data):
        if person is None and year is None:
            params_list.append({"idx": i, "title": title, "kind": "titles"})
        elif year is None:
            params_list.append(
                {
                    "idx": i,
                    "title": title,
                    "authors": person if isinstance(person, list) else [person],
                    "kind": "titles_authors",
                }
            )
        elif person is None:
            params_list.append(
                {"idx": i, "title": title, "year": year, "kind": "titles_year"}
            )

    with pool.connection() as conn:
        with conn.cursor() as cur:
            results = _execute_queries(cur=cur, params_list=params_list)

    return results


def set_sim_threshold(thresh: float, connection_pool: ConnectionPool) -> None:
    """
    Sets the similarity threshold to be used when performing fuzzy queries. Needs to be set each time the
    database is spun up
    Args:
        thresh: The similarity threshold
        connection_pool: The connection pool to use

    Returns:
        None
    """
    with connection_pool.connection() as conn:
        cur = conn.execute(f"SET pg_trgm.similarity_threshold = {thresh}")
        conn.commit()
        cur.close()
