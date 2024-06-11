import asyncio
from typing import Optional, Any

from asyncpg import Pool
from joblib import Parallel, delayed
from loguru import logger
from tqdm.asyncio import tqdm
from nesy.dataset_augmentation import state
from nesy.dataset_augmentation.api_querying.QueryResults import QueryResults
from .execute_queries import execute_queries
from .utils import group_by_query_index, flatmap, set_sim_threshold


def _process_query_results(
    rows: list[list[tuple[str, str, list[str], str]]]
) -> Optional[tuple[str, dict[str, Any]]]:
    # get distances
    distances = [float(row[-1][1]) for row in rows]

    # if there are multiple rows with distance zero, return None
    if distances.count(0) > 1:
        return None

    def _make_return(candidate: list[tuple[str, str, list[str], str]]):
        query_index = candidate[0][1]
        title = candidate[1][1]
        person = candidate[2][1]
        year = candidate[3][1]
        return (
            query_index,
            {"title": title, "person": person, "year": year, "err": None},
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
    params_lists = [[], [], []]
    query_lookup = {}
    for i, (title, person, year) in enumerate(query_data):
        query_lookup[i] = title
        if person is None and year is None:
            params_lists[0].append([f"{i}", title])
        elif year is None:
            params_lists[1].append([f"{i}", title, person])
        elif person is None:
            params_lists[2].append([f"{i}", title, year])

    tasks = []
    if len(params_lists[0]) > 0:
        tasks.append(
            execute_queries(
                params_list=params_lists[0], kind="title", psql_pool=psql_pool
            )
        )
    if len(params_lists[1]) > 0:
        tasks.append(
            execute_queries(
                params_list=params_lists[1], kind="title_authors", psql_pool=psql_pool
            )
        )
    if len(params_lists[2]) > 0:
        tasks.append(
            execute_queries(
                params_list=params_lists[2], kind="title_year", psql_pool=psql_pool
            )
        )

    query_results = await asyncio.gather(*tasks)

    flattened_query_results = flatmap(query_results, f=lambda x: x)
    query_results_by_index = group_by_query_index(flattened_query_results)

    processed_results = Parallel(n_jobs=-1, backend="loky")(
        delayed(_process_query_results)(results) for results in query_results_by_index
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
