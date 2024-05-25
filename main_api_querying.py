import asyncio
import json
import os.path
from typing import Any, Union, Literal

from loguru import logger

from nesy.dataset_augmentation.api_querying.get_movies_and_tv_info import (
    get_movies_and_tv_info,
)
from nesy.dataset_augmentation.api_querying.get_records_info_v2 import get_records_info
from nesy.dataset_augmentation.api_querying.google_kg_search import (
    google_kg_search as get_books_info,
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
    logger.info(
        f"Remaining items: {len(titles)}, processing: {limit if limit <= len(titles) else len(titles)}..."
    )

    if item_type == "movies_and_tv":
        query_fn = get_movies_and_tv_info
    elif item_type == "books":
        query_fn = get_books_info
    elif item_type == "cds_and_vinyl":
        query_fn = get_records_info
    else:
        raise ValueError("Unsupported item type")

    items_info = await query_fn(titles[:limit] if limit != -1 else titles)

    # title is the same used for querying, the one provided by the response is disregarded
    for info in items_info.values():
        asin = items[info["title"]]
        if metadata[asin]["person"] is None:
            metadata[asin]["person"] = info["person"]
        if metadata[asin]["year"] is None:
            metadata[asin]["year"] = info["year"]
        metadata[asin]["queried"] = True


async def main():
    merged_metadata_file_path = os.path.join(
        "data", "processed", "merged_metadata.json"
    )
    merged_metadata_aug_file_path = os.path.join(
        "data", "processed", "merged_metadata_aug.json"
    )

    if not os.path.exists(merged_metadata_aug_file_path):
        with open(merged_metadata_file_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        for k, v in metadata.items():
            metadata[k]["queried"] = False
        with open(merged_metadata_aug_file_path, "w", encoding="utf-8") as g:
            json.dump(metadata, g, indent=4, ensure_ascii=False)

    with open(merged_metadata_aug_file_path, "r+", encoding="utf-8") as g:
        metadata = json.load(g)
        g.seek(0)
        # modify in-place
        await query_apis(metadata, item_type="cds_and_vinyl", limit=5000)
        json.dump(metadata, g, indent=4, ensure_ascii=False)
        g.truncate()


if __name__ == "__main__":
    asyncio.run(main())
