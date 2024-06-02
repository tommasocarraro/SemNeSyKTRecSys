import asyncio
import signal
from typing import Any, Union, Literal, Callable

from loguru import logger
import inspect
from nesy.dataset_augmentation import state
from nesy.dataset_augmentation.api_querying import (
    get_books_info,
    get_records_info,
    get_movies_and_tv_info,
)
from nesy.dataset_augmentation.api_querying.utils import ErrorCode
from nesy.dataset_augmentation.api_querying.open_library import get_books_info


def _is_async_function(func: Callable) -> bool:
    return inspect.iscoroutinefunction(func)


async def _run_query(
    batch,
    item_type: Union[
        Literal["movies_and_tv"], Literal["books"], Literal["cds_and_vinyl"]
    ],
):
    if item_type == "movies_and_tv":
        query_fn = get_movies_and_tv_info
        args = []
    elif item_type == "books":
        query_fn = get_books_info
        args = []
        # query_fn = get_books_info
        # args = [["Book"]]
    elif item_type == "cds_and_vinyl":
        query_fn = get_records_info
        args = []
    else:
        raise ValueError("Unsupported item type")

    if _is_async_function(query_fn):
        return await query_fn(batch, *args)
    else:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, query_fn, batch, *args)


async def query_apis(
    metadata: dict[str, Any],
    item_type: Union[
        Literal["movies_and_tv"], Literal["books"], Literal["cds_and_vinyl"]
    ],
    batch_size: int = -1,
) -> None:
    # attach hooks for signal handling
    def _signal_handler(_sig, _frame):
        state.GRACEFUL_EXIT = True

    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, _signal_handler)

    # dictionary for reverse lookup
    items = {
        v["title"]: (k, v)
        for k, v in metadata.items()
        if v["type"] == item_type
        and v["title"] is not None
        and (v["person"] is None or v["year"] is None)
        and not v["queried"]
    }

    query_data = [
        (item["title"], item["person"], item["year"]) for _, item in items.values()
    ]

    if batch_size == -1 or batch_size > len(query_data):
        batch_size = len(query_data)
    for i in range(0, len(query_data), batch_size):
        logger.info(
            f"Remaining items: {len(query_data) - i}, processing: {batch_size if batch_size <= len(query_data) else len(query_data)}..."
        )
        batch = query_data[i : i + batch_size]
        items_info = await _run_query(batch, item_type)

        logger.info(items_info)
        exit(0)

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
                person = info["person"]
                if metadata[asin]["person"] is None and person is not None:
                    metadata[asin]["person"] = person
                    metadata[asin]["from_api"] = metadata[asin]["from_api"] + ["person"]
                year = info["year"]
                if metadata[asin]["year"] is None and year is not None:
                    metadata[asin]["year"] = year
                    metadata[asin]["from_api"] = metadata[asin]["from_api"] + ["year"]
                metadata[asin]["queried"] = True

        if state.GRACEFUL_EXIT:
            logger.info("Terminating early due to interrupt")
            break

    exit(1)
