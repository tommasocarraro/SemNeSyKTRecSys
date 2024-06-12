from asyncio import CancelledError
from typing import Optional, Any

from asyncpg import Pool
from tqdm.asyncio import tqdm

from .statements import get_statement, reset_statements
from ..utils import _add_signal_handlers, ErrorCode


async def _execute_prepared(
    params: list, kind: str, psql_pool: Pool, how_many_authors: Optional[int] = None
):
    async with psql_pool.acquire() as psql_conn:
        statement = await get_statement(
            kind=kind, how_many_authors=how_many_authors, psql_conn=psql_conn
        )
        return await statement.fetch(*params)


async def execute_queries(
    params_list: list[list[str]], psql_pool: Pool
) -> list[tuple[list[list[tuple[str, str, str, str]]], Optional[ErrorCode]]]:
    _add_signal_handlers()
    tasks = []
    for params in params_list:
        kind = params[0]
        query_index = params[1]
        title = params[2]
        how_many_authors = None
        if kind == "title_authors":
            authors = params[3]
            actual_params = [query_index, title, *authors]
            how_many_authors = len(authors)
        elif kind == "title_year":
            year = params[3]
            actual_params = [query_index, title, year]
        else:
            actual_params = [query_index, title]
        tasks.append(
            _execute_prepared(
                params=actual_params,
                kind=kind,
                psql_pool=psql_pool,
                how_many_authors=how_many_authors,
            )
        )
    results = []
    for response in (
        pbar := tqdm.as_completed(
            tasks,
            dynamic_ncols=True,
            desc="Running queries against Open Library...",
        )
    ):
        data, err = None, None
        try:
            data = await response
            data = [list(x.items()) for x in data]
        except CancelledError:
            pbar.close()
            err = ErrorCode.Cancelled
        finally:
            reset_statements()
            results.append((data, err))
    return results
