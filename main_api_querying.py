import asyncio
import json
import os.path
from typing import Any
from loguru import logger
from nesy.dataset_augmentation.api_querying.get_movies_and_tv_info import (
    get_movies_and_tv_info,
)


async def query_movies_and_tv(metadata: dict[str, Any], limit: int = 500) -> None:
    # dictionary for reverse lookup
    items = {
        v["title"]: k
        for k, v in metadata.items()
        if v["type"] == "movies_and_tv"
        and v["title"] is not None
        and (v["person"] is None or v["year"] is None)
        and not v["queried"]
    }

    titles = list(items.keys())
    logger.info(f"Remaining items: {len(titles)}, processing: {limit}...")

    items_info = await get_movies_and_tv_info(titles=titles[:limit])

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
        await query_movies_and_tv(metadata, 10000)
        json.dump(metadata, g, indent=4, ensure_ascii=False)
        g.truncate()


if __name__ == "__main__":
    asyncio.run(main())
