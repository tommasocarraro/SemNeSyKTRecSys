import json
import os.path
import signal
from asyncio import CancelledError
from typing import Any, Union, Literal, Optional, TextIO

from loguru import logger
from psycopg import OperationalError
from psycopg_pool import AsyncConnectionPool

from config import PSQL_CONN_STRING
from nesy.dataset_augmentation import state
from nesy.dataset_augmentation.api_querying import (
    get_records_info,
    get_movies_and_tv_info,
)
from nesy.dataset_augmentation.api_querying.QueryResults import QueryResults
from nesy.dataset_augmentation.api_querying.open_library import get_books_info
from nesy.dataset_augmentation.api_querying.utils import ErrorCode


async def _run_queries(
    batch,
    item_type: Union[Literal["movies"], Literal["books"], Literal["music"]],
    psql_pool: Optional[AsyncConnectionPool] = None,
) -> dict[str, QueryResults]:
    if item_type == "movies":
        query_fn = get_movies_and_tv_info
        args = []
    elif item_type == "books":
        query_fn = get_books_info
        args = [psql_pool]
    elif item_type == "music":
        query_fn = get_records_info
        args = []
    else:
        raise ValueError("Unsupported item type")

    return await query_fn(batch, *args)


async def query_apis(
    metadata: dict[str, Any],
    item_type: Union[Literal["movies"], Literal["books"], Literal["music"]],
    output_file: TextIO,
    batch_size: int = -1,
) -> None:
    # attach hooks for signal handling
    def _signal_handler(_sig, _frame):
        state.GRACEFUL_EXIT = True

    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, _signal_handler)

    # dictionary for reverse lookup
    items = {
        v["title_cleaned"]: (k, v)
        for k, v in metadata.items()
        if v["type"] == item_type
        and v["title_cleaned"] is not None
        and (v["person"] is None or v["year"] is None)
        and not v["queried"]
    }

    query_data = [
        (item["title_cleaned"], item["person"], item["year"])
        for _, item in items.values()
    ]

    if item_type == "books":
        logger.info("Establishing a new connection pool to PostgreSQL")
        try:
            psql_pool = AsyncConnectionPool(
                conninfo=PSQL_CONN_STRING,
                min_size=12,
                max_size=12,
                num_workers=3,
                open=False,
                timeout=float("inf"),
            )
            await psql_pool.open(wait=True)
        except ConnectionRefusedError as e:
            logger.error(f"Failed to establish a new connection pool: {e}")
            exit(1)
    else:
        psql_pool = None

    if batch_size == -1 or batch_size > len(query_data):
        batch_size = len(query_data)

    for i in range(0, len(query_data), batch_size):
        logger.info(
            f"Remaining items: {len(query_data) - i}, processing: {max(batch_size, len(query_data))}..."
        )
        batch = query_data[i : i + batch_size]

        items_info = await _run_queries(batch, item_type, psql_pool)

        # title is the same used for querying, the one provided by the response is disregarded
        for info in items_info.values():
            title = info["title"]
            asin, _ = items[title]
            err = info["err"]
            if err is not None:
                if err in [ErrorCode.Cancelled, ErrorCode.Throttled]:
                    continue
                else:
                    metadata[asin]["queried"] = True
                    metadata[asin]["err"] = err.name
            else:
                api_name = info["api_name"]
                person = info["person"]
                if metadata[asin]["person"] is None and person is not None:
                    metadata[asin]["person"] = person
                    metadata[asin]["metadata_source"]["person"] = api_name
                year = info["year"]
                if metadata[asin]["year"] is None and year is not None:
                    metadata[asin]["year"] = year
                    metadata[asin]["metadata_source"]["year"] = api_name
                metadata[asin]["queried"] = True

        logger.info(f"Writing updated metadata to {os.path.abspath(output_file.name)}")
        # reset file cursor position so writing the data back will overwrite previous contents
        output_file.seek(0)
        json.dump(metadata, output_file, indent=4, ensure_ascii=False)
        output_file.truncate()

        if state.GRACEFUL_EXIT:
            logger.info("Terminating early due to interrupt")
            try:
                await psql_pool.close(timeout=0)
            except (CancelledError, OperationalError):
                pass
            break
