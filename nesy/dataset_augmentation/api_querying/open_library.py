from typing import Optional

from loguru import logger
from psycopg_pool import ConnectionPool

from config import PSQL_CONN_STRING
from nesy.dataset_augmentation import state

pool = None


def get_books_info(query_data: list[tuple[str, Optional[str], Optional[str]]]):
    global pool
    if pool is None:
        pool = ConnectionPool(PSQL_CONN_STRING)
    pool.wait()
    if not state.SET_SIM_THRESHOLD:
        set_sim_threshold(0.9, pool)
    data = fuzzy_search_titles(query_data, pool)
    return data


def _execute_many_queries(
    cur, query: str, params: list[dict[str, str]]
) -> list[list[tuple]]:
    res = []
    if len(params) > 0:
        cur.executemany(query, params, returning=True)
        while True:
            res.append(cur.fetchall())
            if not cur.nextset():
                break
    return res


def fuzzy_search_titles(
    query_data: list[tuple[str, Optional[str], Optional[str]]], pool: ConnectionPool
) -> list[list[tuple]]:
    """
    Performs an asynchronous fuzzy search against the PostgreSQL database through the provided connection pool
    Args:
        query_data: query data to search for
        pool: The connection pool to use

    Returns:
        The query output
    """
    logger.info(f"Performing {len(query_data)} fuzzy searches against Open Library")
    query_data_only_titles = []
    query_data_titles_authors = []
    query_data_titles_years = []
    for i, query in enumerate(query_data):
        if query[1] is None and query[2] is None:
            query_data_only_titles.append({"idx": i, "title": query[0]})
        elif query[1] is None:
            query_data_titles_authors.append(
                {"idx": i, "title": query[0], "authors": query[1]}
            )
        else:
            query_data_titles_years.append(
                {"idx": i, "title": query[0], "year": query[2]}
            )

    query_only_titles = """
        SELECT %(idx)s, title, authors, year
        FROM edition_authors_materialized_view
        WHERE title %% %(title)s
        ORDER BY (title <-> %(title)s)
        LIMIT 10"""
    query_titles_authors = """
        SELECT %(idx)s, title, authors, year
        FROM edition_authors_materialized_view
        WHERE title %% %(title)s
        AND EXISTS(
            SELECT 1
            FROM unnest(authors) AS author
            WHERE author %% ANY(ARRAY[%(authors)s])
        )
        ORDER BY (title <-> %(title)s)
        LIMIT 10"""
    query_titles_years = """
        SELECT %(idx)s, title, authors, year
        FROM edition_authors_materialized_view
        WHERE title %% %(title)s AND year LIKE %(year)s
        ORDER BY (title <-> %(title)s)
        LIMIT 10"""
    with pool.connection() as conn:
        with conn.cursor() as cur:
            results = []
            results.extend(
                _execute_many_queries(cur, query_only_titles, query_data_only_titles)
            )
            results.extend(
                _execute_many_queries(
                    cur, query_titles_authors, query_data_titles_authors
                )
            )
            results.extend(
                _execute_many_queries(cur, query_titles_years, query_data_titles_years)
            )

    # TODO joblib
    for res in results:
        query_index = res[0][0]
        query = query_data[query_index]
        logger.info(query)
        for r in res:
            _, title, authors, year = r
            logger.info(f"title: {title}, authors: {authors}, year: {year}")
        exit()

    return results


def fuzzy_search_on_works(
    query_data: list[str], pool: ConnectionPool
) -> list[list[tuple]]:
    logger.info(f"Performing {len(query_data)} fuzzy searches against Open Library")

    query_dict = []
    for i, query in enumerate(query_data):
        query_dict.append({"idx": i, "title": query})
    query_only_titles = """
        SELECT %(idx)s, title, authors
        FROM works_authors_materialized_view
        WHERE title %% %(title)s
        ORDER BY (title <-> %(title)s)
        LIMIT 10"""
    with pool.connection() as conn:
        with conn.cursor() as cur:
            results = _execute_many_queries(cur, query_only_titles, query_dict)
    # TODO joblib
    for res in results:
        query_index = res[0][0]
        logger.info(query_data[query_index])
        for r in res:
            _, title, authors = r
            logger.info(f"title: {title}, authors: {authors}")
        exit()

    return results


def set_sim_threshold(thresh: float, pool: ConnectionPool) -> None:
    """
    Sets the similarity threshold to be used when performing fuzzy queries. Needs to be set each time the
    database is spun up
    Args:
        thresh: The similarity threshold
        pool: The connection pool to use

    Returns:
        None
    """
    with pool.connection() as conn:
        # SET commands cannot be used with parameters, so I'm building the query string manually
        cur = conn.execute(f"SET pg_trgm.similarity_threshold = {thresh}")
        conn.commit()
        cur.close()
