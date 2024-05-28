import signal
from typing import Any, Union, Literal

from loguru import logger

from nesy.dataset_augmentation import state
from nesy.dataset_augmentation.api_querying import (
    get_books_info,
    get_records_info,
    get_movies_and_tv_info,
)


async def query_apis(
    metadata: dict[str, Any],
    item_type: Union[
        Literal["movies_and_tv"], Literal["books"], Literal["cds_and_vinyl"]
    ],
    limit: int = -1,
) -> None:
    # attach hooks for signal handling
    def _signal_handler(_sig, _frame):
        state.GRACEFUL_EXIT = True

    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, _signal_handler)

    # dictionary for reverse lookup
    items = {
        v["title"]: k
        for k, v in metadata.items()
        if v["type"] == item_type
        and v["title"] is not None
        and (v["person"] is None or v["year"] is None)
        and not v["queried"]
    }

    titles = list(items.keys())

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

    for i in range(0, len(titles), limit):
        logger.info(
            f"Remaining items: {len(titles) - i}, processing: {limit if limit <= len(titles) else len(titles)}..."
        )
        batch = titles[i : i + limit]
        items_info = await query_fn(batch, *args)

        # title is the same used for querying, the one provided by the response is disregarded
        for info in items_info.values():
            if not info["err"]:
                asin = items[info["title"]]
                if metadata[asin]["person"] is None:
                    metadata[asin]["person"] = info["person"]
                if metadata[asin]["year"] is None:
                    metadata[asin]["year"] = info["year"]
                metadata[asin]["queried"] = True

        if state.GRACEFUL_EXIT:
            break
