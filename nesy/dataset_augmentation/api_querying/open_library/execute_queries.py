from asyncio import CancelledError
from typing import Optional

from psycopg import OperationalError
from psycopg_pool import AsyncConnectionPool
from tqdm.asyncio import tqdm

from .queries import query_title, query_title_authors, query_title_year
from .utils import set_sim_threshold
from ..utils import _add_signal_handlers, ErrorCode

type Row = list[tuple[str, str]]
type Rows = list[Row]
type Params = dict[str, str | list[str]]


async def _execute_query(
    params_with_index: tuple[int, Params], psql_pool: AsyncConnectionPool
) -> Optional[tuple[int, Rows]]:
    query_index, params = params_with_index
    results = None
    try:
        async with psql_pool.connection() as psql_conn:
            await psql_conn.set_autocommit(True)
            async with psql_conn.cursor() as cur:
                # this threshold is session based, so it needs to be set on a per-connection basis
                await set_sim_threshold(0.9, cur)
                if "authors" in params:
                    query = query_title_authors
                elif "year" in params:
                    query = query_title_year
                else:
                    query = query_title
                await cur.execute(query, params)
                if cur.rowcount > 0:
                    cols = [col.name for col in cur.description]
                    rows = await cur.fetchall()
                    results = list(zip(cols, *rows))
    except (CancelledError, OperationalError):
        pass
    finally:
        return query_index, results


async def execute_queries(
    params_list: list[tuple[int, Params]],
    psql_pool: AsyncConnectionPool,
) -> list[tuple[int, Optional[Rows], Optional[ErrorCode]]]:
    _add_signal_handlers()

    tasks = [
        _execute_query(params_with_index=params_with_index, psql_pool=psql_pool)
        for params_with_index in params_list
    ]

    results = []
    for response in (
        pbar := tqdm.as_completed(
            tasks,
            dynamic_ncols=True,
            desc="Running queries against Open Library...",
        )
    ):
        err = None
        try:
            query_index, rows = await response
        except CancelledError:
            pbar.close()
            err = ErrorCode.Cancelled
        except OperationalError:
            pass
        finally:
            results.append((query_index, rows, err))
    return results
