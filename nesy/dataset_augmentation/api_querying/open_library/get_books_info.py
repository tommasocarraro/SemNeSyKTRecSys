from typing import Optional, Any

from asyncpg import Pool
from joblib import Parallel, delayed

from nesy.dataset_augmentation import state
from nesy.dataset_augmentation.api_querying.QueryResults import QueryResults
from .execute_queries import execute_queries
from .utils import set_sim_threshold
from ..utils import ErrorCode


def _process_query_results(
    result: tuple[list[list[tuple[str, str, str, str]]], Optional[ErrorCode]]
) -> Optional[tuple[str, dict[str, Any]]]:
    rows, err = result
    if rows is None:
        return None
    # get distances
    distances = [float(row[-1][1]) for row in rows]

    # if there are multiple rows with distance zero, return None
    if distances.count(0) > 1:
        return None

    def _make_return(candidate: list[tuple[str, str, str, str]]):
        query_index = candidate[0][1]
        title = candidate[1][1]
        person = candidate[2][1]
        year = candidate[3][1]
        return (
            query_index,
            {"title": title, "person": person, "year": year, "err": err},
        )

    # if there is only one row and its distance is at most significant_margin, or if there are more rows and the
    # distance between the second and the first is at least significant_margin, return the first
    significant_margin = 0.2
    if (len(distances) == 1 and distances[0] < significant_margin) or (
        len(distances) > 1 and distances[1] - distances[0] >= significant_margin
    ):
        return _make_return(rows[0])

    return None


async def _prepare_db_queries(
    query_data: list[tuple[str, Optional[str], Optional[str]]], psql_pool: Pool
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
        query_lookup[i] = title
        if person is None and year is None:
            params_list.append(["title", f"{i}", title])
        elif year is None:
            params_list.append(["title_authors", f"{i}", title, person])
        elif person is None:
            params_list.append(["title_year", f"{i}", title, year])

    query_results = await execute_queries(params_list=params_list, psql_pool=psql_pool)

    processed_results = Parallel(n_jobs=-1, backend="loky")(
        delayed(_process_query_results)(results) for results in query_results
    )

    return {
        query_lookup[int(res[0])]: {
            "title": query_lookup[int(res[0])],
            "person": res[1]["person"],
            "year": res[1]["year"],
            "err": res[1]["err"],
            "api_name": "Open Library",
        }
        for res in processed_results
        if res is not None
    }


async def get_books_info(
    query_data: list[tuple[str, Optional[str], Optional[str]]], psql_pool: Pool
) -> dict[str, QueryResults]:
    if not state.SET_SIM_THRESHOLD:
        await set_sim_threshold(0.9, psql_pool)

    return await _prepare_db_queries(query_data, psql_pool)
