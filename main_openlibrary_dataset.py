from asyncio import run
from os.path import join

from nesy.dataset_augmentation.metadata_extraction.open_library import (
    process_dump,
    build_cache,
)


async def main():
    process_dump_flag = False
    build_cache_flag = True

    editions_processed_file_path = join("data", "open_library", "editions.jsonl")
    authors_processed_file_path = join("data", "open_library", "authors.jsonl")
    works_processed_file_path = join("data", "open_library", "works.jsonl")
    if process_dump_flag:
        editions_in_file_path = join(
            "data", "open_library", "ol_dump_editions_2024-05-31.txt"
        )
        authors_in_file_path = join(
            "data", "open_library", "ol_dump_authors_2024-05-31.txt"
        )
        works_in_file_path = join(
            "data", "open_library", "ol_dump_works_2024-05-31.txt"
        )
        process_dump(
            editions_in_file_path=editions_in_file_path,
            editions_out_file_path=editions_processed_file_path,
            authors_in_file_path=authors_in_file_path,
            authors_out_file_path=authors_processed_file_path,
            works_in_file_path=works_in_file_path,
            works_out_file_path=works_processed_file_path,
        )

    if build_cache_flag:
        build_cache(
            editions_dump_path=editions_processed_file_path,
            authors_dump_path=authors_processed_file_path,
            works_dump_path=works_processed_file_path,
        )


if __name__ == "__main__":
    run(main())
