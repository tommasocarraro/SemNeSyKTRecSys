from os.path import join

from loguru import logger
from psycopg_pool import ConnectionPool

from config import PSQL_CONN_STRING
from nesy.dataset_augmentation.api_querying.open_library import (
    set_sim_threshold,
    fuzzy_search_titles,
)
from nesy.dataset_augmentation.metadata_extraction.open_library import (
    process_dump,
    build_cache,
)


def pretty_print_results(results: list[list[tuple]]) -> None:
    for result in results:
        logger.info(result)


def main():
    process_dump_flag = False
    build_cache_flag = False
    query_flag = True

    if process_dump_flag:
        editions_in_file_path = join(
            "data", "open_library", "ol_dump_editions_2024-04-30.txt"
        )
        editions_out_file_path = join("data", "open_library", "editions.jsonl")
        authors_in_file_path = join(
            "data", "open_library", "ol_dump_authors_2024-04-30.txt"
        )
        authors_out_file_path = join("data", "open_library", "authors.jsonl")
        process_dump(
            editions_in_file_path,
            editions_out_file_path,
            authors_in_file_path,
            authors_out_file_path,
        )

    if build_cache_flag:
        authors_dump_path = join("data", "open_library", "authors.jsonl")
        editions_dump_path = join("data", "open_library", "editions.jsonl")
        build_cache(authors_dump_path, editions_dump_path)

    if query_flag:
        set_sim_threshold(0.8)

        titles = [
            "Eragon",
            "The Lord of the Rings",
            "Dracula",
            "Uomini che Odiano le Donne",
            "Il Battesimo del Fuoco",
            "Baptism of Fire",
        ]
        with ConnectionPool(PSQL_CONN_STRING) as pool:
            pool.wait()  # waiting for the pool to start up
            results = fuzzy_search_titles(titles, pool)
            pretty_print_results(results)


if __name__ == "__main__":
    main()
