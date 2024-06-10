from typing import Optional, Any

from asyncpg import Pool, create_pool
from joblib import Parallel, delayed
from loguru import logger

from config import PSQL_CONN_STRING
from nesy.dataset_augmentation import state
from .queries import make_query
from ...query_apis import QueryResults

psql_pool: Optional[Pool] = None


async def _execute_queries(pool: Pool, params_list: list[dict[str, str]]):
    query_results = []
    for params_dict in params_list:
        query = make_query(params_dict)
        query_index = params_dict["idx"]
        title = params_dict["title"]
        authors = params_dict.get("authors", None)
        year = params_dict.get("year", None)
        params = [query_index, title]
        if authors is not None:
            params.append(*authors)
        if year is not None:
            params.append(year)
        async with pool.acquire() as conn:
            statement = await conn.prepare(query)
            res = await statement.fetch(*params)
            query_results.append(res)
    return query_results


def _process_query_results(results: Any) -> dict[str, QueryResults]:
    pass  # TODO


async def _fuzzy_search_titles(
    query_data: list[tuple[str, Optional[str], Optional[str]]], pool: Pool
) -> dict[str, QueryResults]:
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
            params_list.append({"idx": f"{i}", "title": title, "kind": "titles"})
        elif year is None:
            params_list.append(
                {
                    "idx": f"{i}",
                    "title": title,
                    "authors": person if isinstance(person, list) else [person],
                    "kind": "titles_authors",
                }
            )
        elif person is None:
            params_list.append(
                {"idx": f"{i}", "title": title, "year": year, "kind": "titles_year"}
            )

    query_results = await _execute_queries(pool=pool, params_list=params_list)

    joblib_pool = Parallel(n_jobs=-1, backend="loky")
    processed_results = joblib_pool(
        delayed(_process_query_results)(results) for results in query_results
    )

    return {  # TODO
        "title": {
            "title": "title",
            "person": ["person"],
            "year": "year",
            "err": None,
            "api_name": "Open Library",
        }
        for res in processed_results
        if res is not None
    }


async def _set_sim_threshold(thresh: float, connection_pool: Pool) -> None:
    """
    Sets the similarity threshold to be used when performing fuzzy queries. Needs to be set each time the
    database is spun up
    Args:
        thresh: The similarity threshold
        connection_pool: The connection pool to use

    Returns:
        None
    """
    async with connection_pool.acquire() as conn:
        await conn.execute(f"SET pg_trgm.similarity_threshold = {thresh}")


async def get_books_info(
    query_data: list[tuple[str, Optional[str], Optional[str]]]
) -> dict[str, QueryResults]:
    global psql_pool
    if psql_pool is None:
        psql_pool = await create_pool(
            dsn=PSQL_CONN_STRING, max_size=20, max_queries=100000
        )
    if not state.SET_SIM_THRESHOLD:
        await _set_sim_threshold(0.9, psql_pool)
    return await _fuzzy_search_titles(query_data, psql_pool)
