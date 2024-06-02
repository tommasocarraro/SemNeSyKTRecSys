from os.path import join

from loguru import logger
from psycopg_pool import ConnectionPool

from config import PSQL_CONN_STRING
from nesy.dataset_augmentation.api_querying.open_library import (
    set_sim_threshold,
    fuzzy_search_titles,
    fuzzy_search_on_works,
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

    editions_processed_file_path = join("data", "open_library", "editions.jsonl")
    authors_processed_file_path = join("data", "open_library", "authors.jsonl")
    works_processed_file_path = join("data", "open_library", "works.jsonl")
    if process_dump_flag:
        editions_in_file_path = join(
            "data", "open_library", "ol_dump_editions_2024-04-30.txt"
        )
        authors_in_file_path = join(
            "data", "open_library", "ol_dump_authors_2024-04-30.txt"
        )
        works_in_file_path = join(
            "data", "open_library", "ol_dump_works_2024-04-30.txt"
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

    if query_flag:

        titles = [
            # "Eragon",
            # "The Lord of the Rings",
            # "Dracula",
            # "Uomini che Odiano le Donne",
            # "Il Battesimo del Fuoco",
            # "Baptism of Fire",
            "Paul Anderson: The Mightiest Minister"
        ]
        with ConnectionPool(PSQL_CONN_STRING) as pool:
            pool.wait()  # waiting for the pool to start up
            set_sim_threshold(0.8, pool)
            results = fuzzy_search_on_works(titles, pool)
            pretty_print_results(results)


if __name__ == "__main__":
    main()
