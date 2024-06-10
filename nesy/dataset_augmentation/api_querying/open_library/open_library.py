import asyncio
from typing import Optional, Any
from typing import Union, Literal

from asyncpg import Pool, create_pool
from joblib import Parallel, delayed
from loguru import logger

from config import PSQL_CONN_STRING
from nesy.dataset_augmentation import state
from .queries import get_statement
from ...query_apis import QueryResults


async def _execute_prepared(
    psql_pool: Pool,
    params_list: list[list[str]],
    kind: Union[Literal["title"], Literal["title_authors"], Literal["title_year"]],
    how_many_authors: Optional[int] = None,
):
    results = []
    async with psql_pool.acquire() as conn:
        statement = await get_statement(
            kind=kind, psql_conn=conn, how_many_authors=how_many_authors
        )
        for params in params_list:
            tasks = statement.fetch(*params)
            results = await asyncio.gather(tasks)
    return results


async def _execute_queries(
    psql_pool: Pool,
    params_list: list[list[str]],
    kind: Union[Literal["title"], Literal["title_authors"], Literal["title_year"]],
):
    results = []
    if kind == "title_authors":
        params_dict: dict[int, list[list[str]]] = {}
        for params in params_list:
            authors = params[2]
            if isinstance(authors, list):
                how_many_authors = len(authors)
            elif isinstance(authors, str):
                how_many_authors = 1
            else:
                raise TypeError("Wrong type for person")
            if how_many_authors not in params_dict:
                params_dict[how_many_authors] = [params]
            else:
                params_dict[how_many_authors].append(params)
        for k, v in params_dict.items():
            results.append(
                await _execute_prepared(
                    psql_pool=psql_pool, params_list=v, kind=kind, how_many_authors=k
                )
            )
    else:
        results.append(
            await _execute_prepared(
                psql_pool=psql_pool, params_list=params_list, kind=kind
            )
        )
    return results


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
    params_lists = [[], [], []]
    for i, (title, person, year) in enumerate(query_data):
        if person is None and year is None:
            params_lists[0].append([f"{i}", title])
        elif year is None:
            if isinstance(person, list):
                person_l = person
            else:
                person_l = [person]
            params_lists[1].append([f"{i}", title, *person_l])
        elif person is None:
            params_lists[2].append([f"{i}", title, year])

    tasks = [
        _execute_queries(psql_pool=pool, params_list=params_lists[0], kind="title"),
        _execute_queries(
            psql_pool=pool, params_list=params_lists[1], kind="title_authors"
        ),
        _execute_queries(
            psql_pool=pool, params_list=params_lists[2], kind="title_year"
        ),
    ]
    query_results = await asyncio.gather(*tasks)

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
    psql_pool = await create_pool(dsn=PSQL_CONN_STRING, max_size=20, max_queries=100000)
    if not state.SET_SIM_THRESHOLD:
        await _set_sim_threshold(0.9, psql_pool)
    books_info = await _fuzzy_search_titles(query_data, psql_pool)
    await psql_pool.close()
    return books_info
