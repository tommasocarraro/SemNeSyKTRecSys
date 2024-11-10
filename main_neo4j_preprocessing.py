from src.paths.dataset_preprocessing import (
    create_csv_files_neo4j,
)

if __name__ == "__main__":
    create_csv_files_neo4j(
        "./data/wikidata/wikidata_triples.csv",
        "./data/wikidata/wikidata_labels.csv",
        selected_properties="./data/wikidata/useful-properties.csv",
    )
