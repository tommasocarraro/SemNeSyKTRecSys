import asyncio
import json
import os.path
from typing import Union, Literal

from nesy.dataset_augmentation.api_querying import google_kg_search


async def main():
    merged_metadata_path = os.path.join("data", "processed", "merged_metadata.json")
    merged_metadata_aug = os.path.join("data", "processed", "merged_metadata_aug.json")

    # how many items to process
    limit = 500
    query_type: Union[
        Literal["books"], Literal["cds_and_vinyl"], Literal["movies_and_tv"]
    ] = "movies_and_tv"
    with open(merged_metadata_path, "r") as f:
        with open(merged_metadata_aug, "w", encoding="UTF-8") as g:
            merged_metadata = json.load(f)

            # dictionary for reverse lookup
            items = {}
            for k, v in merged_metadata.items():
                if v["type"] == query_type and v["title"] is not None:
                    items[v["title"]] = k

            titles = list(items.keys())
            print(f"Remaining books: {len(titles)}, processing: {limit}...")

            items_info = await google_kg_search(
                titles=list(items.keys())[:limit], query_type=query_type
            )

            # title is the same used for querying, the one provided by the response is disregarded
            for title, author, year in items_info:
                asin = items[title]
                merged_metadata[asin]["person"] = author
                merged_metadata[asin]["year"] = year

            json.dump(merged_metadata, g, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    asyncio.run(main())
