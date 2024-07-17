from nesy.paths.neo4j.dataset_preprocessing import create_csv_files_neo4j_no_inverse

if __name__ == "__main__":
    create_csv_files_neo4j_no_inverse("./data/wikidata/claims.wikibase-item_preprocessed.csv",
                                      "./data/wikidata/labels.en.csv",
                                      selected_properties="./data/wikidata/useful-properties.csv")
