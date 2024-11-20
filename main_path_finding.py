from src.paths.FilePaths import FilePaths
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

    # dictionary containing the file paths for both mappings and reviews for each domain, also the popularity threshold
    # to be used when selecting the items from which paths should start
    domains = {
        "movies": {
            "mapping_file_path": "data/processed/mappings/mapping-movies.json",
            "reviews_file_path": "data/processed/legacy/reviews_Movies_and_TV_5.csv",
            "pop_threshold": 300,
        },
        "music": {
            "mapping_file_path": "data/processed/mappings/mapping-music.json",
            "reviews_file_path": "data/processed/legacy/reviews_CDs_and_Vinyl_5.csv",
            "pop_threshold": 300,
        },
        "books": {
            "mapping_file_path": "data/processed/mappings/mapping-books.json",
            "reviews_file_path": "data/processed/legacy/reviews_Books_5.csv",
            "pop_threshold": 300,
        },
    }

    # list of domain pairs for which we want to find paths
    domain_pairs = [
        ("movies", "music"),
        ("movies", "books"),
        ("books", "movies"),
        ("books", "music"),
        ("music", "books"),
        ("music", "movies"),
    ]

    for source_name, target_name in domain_pairs:
        file_paths = FilePaths(
            source_domain_name=source_name,
            mapping_source_domain=domains[source_name]["mapping_file_path"],
            reviews_source_domain=domains[source_name]["reviews_file_path"],
            target_domain_name=target_name,
            mapping_target_domain=domains[target_name]["mapping_file_path"],
            reviews_target_domain=domains[target_name]["reviews_file_path"],
        )
        neo4j_path_finder(
            database_name=database_name,
            file_paths=file_paths,
            max_hops=10,
            n_threads=n_threads,
            cs_threshold=5,
            pop_threshold=domains[source_name]["pop_threshold"],
        )

    should_export = True
    if should_export:
        dataset_export(
            database_name=database_name,
            export_dir_path="data/processed/paths/",
            domains_pairs=domain_pairs,
        )
