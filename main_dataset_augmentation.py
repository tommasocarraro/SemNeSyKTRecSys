import os

from nesy.dataset_augmentation.build_sqlite_cache import build_sqlite_cache
from nesy.dataset_augmentation.extract_metadata import extract_metadata
from nesy.dataset_augmentation.merge_metadata_for_wikidata import (
    merge_metadata_for_wikidata,
)

file_paths = [
    os.path.join("data", "amazon2023", "meta_Books.jsonl"),
    os.path.join("data", "amazon2023", "meta_CDs_and_Vinyl.jsonl"),
    os.path.join("data", "amazon2023", "meta_Movies_and_TV.jsonl"),
]
cache_path = os.path.join("data", "amazon2023", "meta_cache.sqlite3.db")
cache_fail_path = os.path.join("data", "amazon2023", "meta_cache_failures.jsonl")
# build_sqlite_cache(file_paths, cache_path, cache_fail_path)

asin_file_path = os.path.join("data", "processed", "filtered-metadata.json")
metadata_extraction_failure_path = os.path.join(
    "data", "processed", "failed-metadata.txt"
)
extracted_metadata = extract_metadata(asin_file_path, file_paths, cache_path)

merged_output_path = os.path.join("data", "processed", "merged_metadata.json")
merge_metadata_for_wikidata(extracted_metadata, merged_output_path)