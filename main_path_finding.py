from src.paths.dataset_export import dataset_export
from src.paths.dataset_import import dataset_import
from src.paths.path_finder import neo4j_path_finder

if __name__ == "__main__":
    database_name = "wikidata"
    should_import = True
    if should_import:
        dataset_import(
            database_name=database_name,
            nodes_file_path="data/wikidata/nodes.csv",
            rels_file_path="data/wikidata/relationships.csv",
            use_sudo=True,
        )

    n_threads = 12

    neo4j_path_finder(
        database_name=database_name,
        mapping_file_1="data/processed/mappings/mapping-movies.json",
        mapping_file_2="data/processed/mappings/mapping-music.json",
        max_hops=10,
        n_threads=n_threads,
        cold_start=True,
        popular=True,
        cs_threshold=5,
        pop_threshold=300,
    )
    neo4j_path_finder(
        database_name=database_name,
        mapping_file_1="data/processed/mappings/mapping-movies.json",
        mapping_file_2="data/processed/mappings/mapping-books.json",
        max_hops=10,
        n_threads=n_threads,
        cold_start=True,
        popular=True,
        cs_threshold=5,
        pop_threshold=300,
    )
    neo4j_path_finder(
        database_name=database_name,
        mapping_file_1="data/processed/mappings/mapping-books.json",
        mapping_file_2="data/processed/mappings/mapping-movies.json",
        max_hops=10,
        n_threads=n_threads,
        cold_start=True,
        popular=True,
        cs_threshold=5,
        pop_threshold=300,
    )
    neo4j_path_finder(
        database_name=database_name,
        mapping_file_1="data/processed/mappings/mapping-books.json",
        mapping_file_2="data/processed/mappings/mapping-music.json",
        max_hops=10,
        n_threads=n_threads,
        cold_start=True,
        popular=True,
        cs_threshold=5,
        pop_threshold=300,
    )
    neo4j_path_finder(
        database_name=database_name,
        mapping_file_1="data/processed/mappings/mapping-music.json",
        mapping_file_2="data/processed/mappings/mapping-books.json",
        max_hops=10,
        n_threads=n_threads,
        cold_start=True,
        popular=True,
        cs_threshold=5,
        pop_threshold=300,
    )
    neo4j_path_finder(
        database_name=database_name,
        mapping_file_1="data/processed/mappings/mapping-music.json",
        mapping_file_2="data/processed/mappings/mapping-movies.json",
        max_hops=10,
        n_threads=n_threads,
        cold_start=True,
        popular=True,
        cs_threshold=5,
        pop_threshold=300,
    )

    should_export = True
    if should_export:
        domains = [
            ("movies", "music"),
            ("movies", "books"),
            ("books", "movies"),
            ("books", "music"),
            ("music", "books"),
            ("music", "movies"),
        ]
        dataset_export(
            database_name=database_name,
            export_dir_path="data/processed/paths/",
            domains=domains,
        )
