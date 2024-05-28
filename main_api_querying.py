import asyncio
import json
import os.path
from typing import Any, Union, Literal

from loguru import logger

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
        items_info, graceful_exit = await query_fn(batch, *args)

        # title is the same used for querying, the one provided by the response is disregarded
        for info in items_info.values():
            if not info["err"]:
                asin = items[info["title"]]
                if metadata[asin]["person"] is None:
                    metadata[asin]["person"] = info["person"]
                if metadata[asin]["year"] is None:
                    metadata[asin]["year"] = info["year"]
                metadata[asin]["queried"] = True

        if graceful_exit:
            break


async def main():
    merged_metadata_file_path = os.path.join(
        "data", "processed", "merged_metadata.json"
    )
    merged_metadata_aug_file_path = os.path.join(
        "data", "processed", "merged_metadata_aug.json"
    )

    # if the aug file doesn't exist, create it by copying over all the data from the base file, adding queried field
    if not os.path.exists(merged_metadata_aug_file_path):
        with open(merged_metadata_file_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        for k, v in metadata.items():
            metadata[k]["queried"] = False
        with open(merged_metadata_aug_file_path, "w", encoding="utf-8") as g:
            json.dump(metadata, g, indent=4, ensure_ascii=False)

    # set this to True if you want to requery previously queried items
    reset_queried = False

    with open(merged_metadata_aug_file_path, "r+", encoding="utf-8") as g:
        metadata = json.load(g)
        if reset_queried:
            for k, v in metadata.items():
                metadata[k]["queried"] = False
        # reset file cursor position so writing the data back will overwrite previous contents
        g.seek(0)

        # modify metadata in-place
        # await query_apis(metadata, item_type="cds_and_vinyl", limit=5000)
        # await query_apis(metadata, item_type="movies_and_tv", limit=5000)
        await query_apis(metadata, item_type="books", limit=50)

        json.dump(metadata, g, indent=4, ensure_ascii=False)
        g.truncate()


if __name__ == "__main__":
    asyncio.run(main())
