from typing import Optional, Any

from joblib import Parallel, delayed
from loguru import logger
from psycopg import cursor
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


def _make_title_authors_queries(params: list[dict[str, Any]]) -> list[str]:
    def _make_query(title: str, authors: list[str], idx: str):
        from_clauses = ""
        where_clauses = ""
        group_by_clauses = ""
        authors_select = ""
        distance_select = ""

        for i, author in enumerate(authors):
            if from_clauses == "":
                from_clauses = f"combined_materialized_view c{i}"
            else:
                from_clauses += (
                    f" JOIN combined_materialized_view c{i} ON c0.key = c{i}.key"
                )
            if where_clauses == "":
                where_clauses = f"c{i}.title % LOWER('{title}') AND c{i}.author_name % LOWER('{author}')"
            else:
                where_clauses += f" AND c{i}.author_name % LOWER('{author}')"
            if group_by_clauses == "":
                group_by_clauses = f"c{i}.title, c{i}.author_name"
            else:
                group_by_clauses += f", c{i}.author_name"
            if authors_select == "":
                authors_select = f"c{i}.author_name"
            else:
                authors_select += f" || ', ' || c{i}.author_name"
            if i == len(authors) - 1:
                authors_select += " AS authors"

            if distance_select == "":
                distance_select = f"(c{i}.title <-> LOWER('{title}')) + (c{i}.author_name <-> LOWER('{author}'))"
            else:
                distance_select += f" + (c{i}.author_name <-> LOWER('{author}')"
            if i == len(authors) - 1:
                where_clauses += f" AND {distance_select} < 0.5"
                distance_select += " AS distance"
        select_clauses = f"{idx} as query_index, c0.title, {authors_select}, MIN(c0.year) AS year, {distance_select}"

        query = f"""
            SELECT
                {select_clauses}
            FROM
                {from_clauses}
            WHERE
                {where_clauses}
            GROUP BY {group_by_clauses}
            ORDER BY distance
            LIMIT 10
        """

        return query

    return [
        _make_query(params_dict["title"], params_dict["authors"], params_dict["idx"])
        for params_dict in params
    ]


def make_title_year_queries(params: list[dict[str, Any]]) -> list[str]:
    def _make_query(title: str, year: str, idx: str):
        query = f"""
            SELECT
                {idx} AS query_index,
                title,
                STRING_AGG(author_name, ', ') as authors,
                year,
                (title <-> LOWER('{title}')) as distance
            FROM combined_materialized_view
            WHERE
                title % LOWER('{title}')
                AND title <-> LOWER(%(title)s) < 0.5
            AND year ilike '{year}'
            GROUP BY title, year
            ORDER BY distance
            LIMIT 10
        """
        return query

    return [
        _make_query(params_dict["title"], params_dict["year"], params_dict["idx"])
        for params_dict in params
    ]


def make_title_query():
    return """
    SELECT
        %(idx)s AS query_index,
        title,
        MIN(year) as year,
        STRING_AGG(DISTINCT author_name, ', ') as authors,
        title <-> LOWER(%(title)s) as distance
    FROM combined_materialized_view
    WHERE 
        title %% LOWER(%(title)s)
        AND title <-> LOWER(%(title)s) < 0.5
    GROUP BY key, title
    ORDER BY distance
    LIMIT 10
    """


def _execute_titles_only_queries(
    cur: cursor, params: list[dict[str, str]], joblib_pool: Parallel
):
    query = make_title_query()
    cur.executemany(query, params, returning=True)
    query_results = []
    if len(params) == 0:
        return query_results

    while True:
        query_results.append(cur.fetchall())
        if not cur.nextset():
            break

    def _process_result(result: tuple[str, str, float, str, str]):
        return result

    return joblib_pool(delayed(_process_result)(result) for result in query_results)


def _execute_titles_authors_queries(
    cur: cursor, params: list[dict[str, str]], joblib_pool: Parallel
):
    queries = _make_title_authors_queries(params)
    query_results = []
    if len(params) == 0:
        return query_results
    for query in queries:
        cur.execute(query)
        query_results.append(cur.fetchall())

    def _process_result(result: tuple[tuple, tuple]):
        return result

    return joblib_pool(delayed(_process_result)(result) for result in query_results)


def _execute_titles_year_queries(
    cur: cursor, params: list[dict[str, str]], joblib_pool: Parallel
):
    queries = make_title_year_queries(params)
    query_results = []
    if len(params) == 0:
        return query_results
    for query in queries:
        cur.execute(query)
        query_results.append(cur.fetchall())

    def _process_result(result: tuple[str, str, float, str, str]):
        return result

    return joblib_pool(delayed(_process_result)(result) for result in query_results)


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
    query_data_only_titles = []
    query_data_titles_authors = []
    query_data_titles_years = []
    for i, (title, person, year) in enumerate(query_data):
        if person is None and year is None:
            query_data_only_titles.append({"idx": i, "title": title})
        elif year is None:
            query_data_titles_authors.append(
                {
                    "idx": i,
                    "title": title,
                    "authors": person if isinstance(person, list) else [person],
                }
            )
        elif person is None:
            query_data_titles_years.append({"idx": i, "title": title, "year": year})

    with pool.connection() as conn:
        with conn.cursor() as cur:
            with Parallel(n_jobs=-1, backend="loky") as joblib_pool:
                results = []
                results.extend(
                    _execute_titles_only_queries(
                        cur, query_data_only_titles, joblib_pool
                    )
                )
                results.extend(
                    _execute_titles_authors_queries(
                        cur, query_data_titles_authors, joblib_pool
                    )
                )
                results.extend(
                    _execute_titles_year_queries(
                        cur, query_data_titles_years, joblib_pool
                    )
                )

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
