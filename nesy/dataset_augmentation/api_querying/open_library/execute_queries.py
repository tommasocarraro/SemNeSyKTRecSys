from typing import Optional
from typing import Union, Literal

from asyncpg import Pool
from tqdm.asyncio import tqdm

from .statements import get_statement, reset_statements
from .utils import flatmap


async def _execute_prepared(
    params: list,
    kind: Union[Literal["title"], Literal["title_authors"], Literal["title_year"]],
    psql_pool: Pool,
    how_many_authors: Optional[int] = None,
):
    async with psql_pool.acquire() as psql_conn:
        statement = await get_statement(
            kind=kind, how_many_authors=how_many_authors, psql_conn=psql_conn
        )
        return await statement.fetch(*params)


async def execute_queries(
    params_list: list[list[str]],
    kind: Union[Literal["title"], Literal["title_authors"], Literal["title_year"]],
    psql_pool: Pool,
):
    tasks = []
    if kind == "title_authors":
        params_dict: dict[int, list[list[str]]] = {}
        for params in params_list:
            query_index = params[0]
            title = params[1]
            authors = params[2]
            if isinstance(authors, list):
                how_many_authors = len(authors)
            elif isinstance(authors, str):
                how_many_authors = 1
            else:
                raise TypeError("Wrong type for person")
            params = [query_index, title, *authors]
            if how_many_authors not in params_dict:
                params_dict[how_many_authors] = [params]
            else:
                params_dict[how_many_authors].append(params)
        for k, v in params_dict.items():
            for params in v:
                tasks.append(
                    _execute_prepared(
                        params=params,
                        kind=kind,
                        how_many_authors=k,
                        psql_pool=psql_pool,
                    )
                )
    else:
        for params in params_list:
            tasks.append(
                _execute_prepared(params=params, kind=kind, psql_pool=psql_pool)
            )
    results = await tqdm.gather(*tasks, dynamic_ncols=True, desc="Running queries...")
    reset_statements()
    return flatmap(results=results, f=lambda x: list(x.items()))
