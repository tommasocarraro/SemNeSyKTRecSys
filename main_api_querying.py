import asyncio
import json
import os
import os.path
from nesy.dataset_augmentation import query_apis


async def main():
    merged_metadata_file_path = os.path.join(
        "data", "processed", "merged-metadata.json"
    )
    merged_metadata_aug_file_path = os.path.join(
        "data", "processed", "merged-metadata-aug.json"
    )

    # if the aug file doesn't exist, create it by copying over all the data from the base file, adding queried field
    if not os.path.exists(merged_metadata_aug_file_path):
        with open(merged_metadata_file_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        for k, v in metadata.items():
            metadata[k]["queried"] = False
            metadata[k]["err"] = None
        with open(merged_metadata_aug_file_path, "w", encoding="utf-8") as g:
            json.dump(metadata, g, indent=4, ensure_ascii=False)

    # set this to True if you want to requery previously queried items
    retry = False

    with open(merged_metadata_aug_file_path, "r+", encoding="utf-8") as g:
        metadata = json.load(g)
        if retry:
            for k, v in metadata.items():
                metadata[k]["queried"] = False
                metadata_source = metadata[k]["metadata_source"]
                for field, value in metadata_source.items():
                    if metadata_source[field] != "Amazon dataset":
                        metadata_source[field] = None

        # await query_apis(metadata, item_type="music", batch_size=500)
        # await query_apis(metadata, item_type="movies", batch_size=20000)
        await query_apis(metadata, item_type="books", batch_size=100, output_file=g)


if __name__ == "__main__":
    asyncio.run(main())
