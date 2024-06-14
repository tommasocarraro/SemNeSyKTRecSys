from asyncio import CancelledError
from typing import Optional, Union

from psycopg_pool import AsyncConnectionPool
from tqdm.asyncio import tqdm

from .utils import set_sim_threshold
from ..utils import _add_signal_handlers, ErrorCode

query_title_authors = f"""
    WITH author_names AS (
        SELECT unnest(ARRAY[%(authors)s]) AS author_name
    )
    SELECT 
        %(query_index)s AS query_index,
        MIN(year) AS year
    FROM (
        SELECT
           year,
           ea.title_query <-> LOWER(%(title)s) AS title_distance,
           SUM(ea.name_query <-> an.author_name) OVER () AS author_distance
        FROM editions_authors ea
        JOIN author_names an ON ea.name_query %% an.author_name
        WHERE ea.title_query %% LOWER(%(title)s)
        ORDER BY title_distance, author_distance
    ) sub;
"""


query_title_year = f"""
    SELECT 
        %(query_index)s AS query_index,
        string_agg(DISTINCT name, ', ') AS authors
    FROM editions_authors
    WHERE
        title_query %% LOWER(%(title)s)
        AND year = %(year)s
    GROUP BY key, title_query, year;
"""


query_title = f"""
SELECT
    %(query_index)s AS query_index,
    author_names AS authors,
    book_year AS year
FROM get_book_info_by_title(LOWER(%(title)s))
"""


async def _execute_prepared(
    params: dict[str, Union[str, list[str]]], kind: str, psql_pool: AsyncConnectionPool
):
    async with psql_pool.connection() as psql_conn:
        async with psql_conn.cursor() as cur:
            await set_sim_threshold(0.9, cur)
            if kind == "title_authors":
                query = query_title_authors
            elif kind == "title_year":
                query = query_title_year
            else:
                query = query_title
            await cur.execute(query, params)
            if cur.rowcount > 0:
                results = await cur.fetchall()
            else:
                results = None
            return results


async def execute_queries(
    params_list: list[list[str]], psql_pool: AsyncConnectionPool
) -> list[tuple[list[list[tuple[str, str, str, str]]], Optional[ErrorCode]]]:
    _add_signal_handlers()
    tasks = []

    for params in params_list:
        kind = params[0]
        query_index = params[1]
        title = params[2]
        if kind == "title_authors":
            authors = params[3]
            actual_params = {
                "query_index": query_index,
                "title": title,
                "authors": authors,
            }
        elif kind == "title_year":
            year = params[3]
            actual_params = {"query_index": query_index, "title": title, "year": year}
        else:
            actual_params = {"query_index": query_index, "title": title}
        tasks.append(
            _execute_prepared(params=actual_params, kind=kind, psql_pool=psql_pool)
        )

    results = await tqdm.gather(*tasks, desc="Executing queries")

    # results = []
    # for response in (
    #     pbar := tqdm.as_completed(
    #         tasks,
    #         dynamic_ncols=True,
    #         desc="Running queries against Open Library...",
    #     )
    # ):
    #     data, err = None, None
    #     try:
    #         data = await response
    #         data = [list(x.items()) for x in data]
    #     except CancelledError:
    #         pbar.close()
    #         err = ErrorCode.Cancelled
    #     finally:
    #         results.append((data, err))
    return results
