from typing import Optional

from joblib import Parallel, delayed
from psycopg_pool import AsyncConnectionPool

from src.dataset_augmentation.api_querying.QueryResults import QueryResults
from .execute_queries import execute_queries, Rows
from ..utils import ErrorCode


def _process_query_results(
    result: tuple[int, Optional[Rows], Optional[ErrorCode]]
) -> tuple[int, Optional, Optional[ErrorCode]]:
    query_index, rows, err = result
    res = {}
    if rows is not None:
        res = {col_name: value for (col_name, value) in rows}
    else:
        res["person"] = None
        res["year"] = None

    return query_index, res, err


async def _prepare_db_queries(
    query_data: list[tuple[str, Optional[str], Optional[str]]],
    psql_pool: AsyncConnectionPool,
) -> dict[str, QueryResults]:
    """
    Performs an asynchronous fuzzy search against the PostgreSQL database through the provided connection pool
    Args:
        query_data: query data to search for
        psql_pool: the postgresql connection pool

    Returns:
        A dictionary containing metadata from the successful queries
    """
    params_list = []
    query_lookup = {}
    for i, (title, person, year) in enumerate(query_data):
        query_lookup[i] = {"title": title}
        if person is None and year is None:
            params_list.append((i, {"title": title}))
        elif year is None:
            params_list.append((i, {"title": title, "authors": person}))
            query_lookup[i]["person"] = person
        elif person is None:
            params_list.append((i, {"title": title, "year": year}))
            query_lookup[i]["year"] = year

    try:
        query_results = await execute_queries(
            params_list=params_list, psql_pool=psql_pool
        )
    except KeyboardInterrupt:
        pass

    processed_results = Parallel(n_jobs=-1, backend="loky")(
        delayed(_process_query_results)(results) for results in query_results
    )

    results = {}
    for res in processed_results:
        if res is None:
            continue
        query_index, row, err = res
        query = query_lookup[int(query_index)]
        queried_title = query["title"]
        results[queried_title] = row
        if "person" not in results[queried_title]:
            results[queried_title]["person"] = query["person"]
        if "year" not in results[queried_title]:
            results[queried_title]["year"] = query["year"]
        results[queried_title]["title"] = queried_title
        results[queried_title]["api_name"] = "Open Library"
        results[queried_title]["err"] = err

    return results


async def get_books_info(
    query_data: list[tuple[str, Optional[str], Optional[str]]],
    psql_pool: AsyncConnectionPool,
) -> dict[str, QueryResults]:
    return await _prepare_db_queries(query_data, psql_pool)
