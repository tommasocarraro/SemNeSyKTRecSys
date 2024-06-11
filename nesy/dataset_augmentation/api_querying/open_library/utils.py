from collections import defaultdict
from typing import Callable, Iterable

from asyncpg import Pool


def group_by_query_index(results: list[list]):
    grouped_data = defaultdict(list)

    for subresult in results:
        query_index = subresult[0][1]
        grouped_data[query_index].append(subresult)

    return list(grouped_data.values())


def flatmap(results: Iterable[Iterable], f: Callable):
    return [f(res) for sublist in results for res in sublist]


async def set_sim_threshold(thresh: float, psql_pool: Pool) -> None:
    """
    Sets the similarity threshold to be used when performing fuzzy queries. Needs to be set each time the
    database is spun up
    Args:
        thresh: The similarity threshold
        psql_pool: The PostgreSQL connection

    Returns:
        None
    """
    async with psql_pool.acquire() as conn:
        await conn.execute(f"SET pg_trgm.similarity_threshold = {thresh}")
