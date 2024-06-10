import os

from nesy.dataset_augmentation.metadata_extraction.amazon_dataset import (
    merge_metadata_for_wikidata,
    extract_metadata,
)


def main():
    file_paths = [
        os.path.join("data", "amazon2023", "meta_Books.jsonl"),
        os.path.join("data", "amazon2023", "meta_CDs_and_Vinyl.jsonl"),
        os.path.join("data", "amazon2023", "meta_Movies_and_TV.jsonl"),
    ]
    cache_path = os.path.join("data", "amazon2023", "meta_cache.sqlite3.db")
    # build_sqlite_cache(file_paths, cache_path)

    asin_file_path = os.path.join("data", "processed", "filtered-metadata.json")
    extracted_metadata = extract_metadata(asin_file_path, file_paths, cache_path)

    merged_output_path = os.path.join("data", "processed", "merged_metadata.json")
    merge_metadata_for_wikidata(extracted_metadata, merged_output_path)


if __name__ == "__main__":
    main()
