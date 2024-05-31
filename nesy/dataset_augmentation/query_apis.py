import signal
from typing import Any, Union, Literal, Callable

from loguru import logger

from nesy.dataset_augmentation import state
from nesy.dataset_augmentation.api_querying import (
    get_books_info,
    get_records_info,
    get_movies_and_tv_info,
)
from nesy.dataset_augmentation.api_querying.utils import ErrorCode


def _get_query_fn(
    item_type: Union[
        Literal["movies_and_tv"], Literal["books"], Literal["cds_and_vinyl"]
    ]
) -> tuple[Callable, list]:
    if item_type == "movies_and_tv":
        query_fn = get_movies_and_tv_info
        args = []
    elif item_type == "books":
        query_fn = get_books_info
        args = [["Book"]]
    elif item_type == "cds_and_vinyl":
        query_fn = get_records_info
        args = []
    else:
        raise ValueError("Unsupported item type")
    return query_fn, args


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

    query_fn, args = _get_query_fn(item_type)
    if batch_size == -1:
        batch_size = len(query_data)
    for i in range(0, len(query_data), batch_size):
        logger.info(
            f"Remaining items: {len(query_data) - i}, processing: {batch_size if batch_size <= len(query_data) else len(query_data)}..."
        )
        batch = query_data[i : i + batch_size]
        items_info = await query_fn(batch, *args)

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
                if metadata[asin]["person"] is None:
                    metadata[asin]["person"] = info["person"]
                if metadata[asin]["year"] is None:
                    metadata[asin]["year"] = info["year"]
                metadata[asin]["queried"] = True

        if state.GRACEFUL_EXIT:
            logger.info("Terminating early due to interrupt")
            break
