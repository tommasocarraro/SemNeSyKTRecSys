from src.paths.dataset_preprocessing import create_csv_files_neo4j

if __name__ == "__main__":
    # process_wikidata_dump(
    #     input_file_path="", output_labels_file_path="", output_triples_file_path=""
    # )

    create_csv_files_neo4j(
        triples_file_path="./data/wikidata/wikidata_triples.csv",
        labels_file_path="./data/wikidata/wikidata_labels.csv",
        selected_properties="./data/wikidata/selected_properties.csv",
    )
