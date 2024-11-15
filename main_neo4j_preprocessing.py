from typing import Any

from src.paths.dataset_preprocessing import (
    create_csv_files_neo4j,
    process_wikidata_dump,
)
from os import path
import orjson
from loguru import logger

if __name__ == "__main__":
    # process_wikidata_dump(
    #     input_file_path="", output_labels_file_path="", output_triples_file_path=""
    # )

    logger.info("Loading mappings...")
    mapping = {}
    mappings_dir = path.join("data", "processed", "mappings")
    mappings_file_paths = [
        path.join(mappings_dir, "mapping-books.json"),
        path.join(mappings_dir, "mapping-movies.json"),
        path.join(mappings_dir, "mapping-music.json"),
    ]
    for mappings_file_path in mappings_file_paths:
        with open(mappings_file_path, "rb") as f:
            m = orjson.loads(f.read())
            m_inv = {
                item["wiki_id"]: asin
                for (asin, item) in m.items()
                if isinstance(item, dict)
            }
            mapping.update(m_inv)

    create_csv_files_neo4j(
        triples_file_path="./data/wikidata/wikidata_triples.csv",
        labels_file_path="./data/wikidata/wikidata_labels.csv",
        mapping=mapping,
        selected_properties="./data/wikidata/selected_properties.csv",
    )
