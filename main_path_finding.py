from src.paths.FilePaths import FilePaths
from src.paths.dataset_export import dataset_export
from src.paths.dataset_import import dataset_import
from src.paths.path_finder import neo4j_path_finder
from loguru import logger

if __name__ == "__main__":
    database_name = "wikidata"
    should_import = False
    if should_import:
        dataset_import(
            database_name=database_name,
            nodes_file_path="data/wikidata/nodes.csv",
            rels_file_path="data/wikidata/relationships.csv",
            use_sudo=True,
        )

    # dictionary containing the file paths for both mappings and reviews for each domain, also the popularity threshold
    # to be used when selecting the items from which paths should start
    domains = {
        "movies": {
            "mapping_file_path": "data/processed/mappings/mapping-movies.json",
            "reviews_file_path": "data/ratings/reviews_Movies_and_TV_5.csv",
        },
        "music": {
            "mapping_file_path": "data/processed/mappings/mapping-music.json",
            "reviews_file_path": "data/ratings/reviews_CDs_and_Vinyl_5.csv",
        },
        "books": {
            "mapping_file_path": "data/processed/mappings/mapping-books.json",
            "reviews_file_path": "data/ratings/reviews_Books_5.csv",
        },
    }

    # list of domain pairs for which we want to find paths
    domain_pairs = [
        {
            "source": "movies",
            "target": "music",
            "pop_threshold": 200,
            "cs_threshold": 10,
        },
        {
            "source": "books",
            "target": "movies",
            "pop_threshold": 200,
            "cs_threshold": 10,
        },
        {
            "source": "music",
            "target": "movies",
            "pop_threshold": 200,
            "cs_threshold": 10,
        },
    ]

    for pair_dict in domain_pairs:
        source_name = pair_dict["source"]
        target_name = pair_dict["target"]
        logger.info(f"Looking for paths from {source_name} to {target_name}")
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
            max_hops=4,
            n_threads=80,
            cs_threshold=pair_dict["cs_threshold"],
            pop_threshold=pair_dict["pop_threshold"],
        )

    should_export = True
    if should_export:
        dataset_export(
            database_name=database_name,
            export_dir_path="data/kg_paths/",
            domain_pairs=domain_pairs,
            mappings_file_paths=domains,
        )
