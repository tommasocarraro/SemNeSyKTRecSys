import json
import os.path

from .query_by_asin import query_by_asin


def extract_metadata(
    asin_file: str,
    metadata_files: list[str],
    cache_path: str,
    output_path: str,
    failure_path: str,
) -> None:
    """
    For each ASIN code found in asin_file, extract the respective metadata from the SQLITE database.
    Args:
        asin_file: File containing the ASIN codes.
        metadata_files: List of metadata files.
        cache_path: Path to the SQLITE database.
        output_path: Output path.
        failure_path: Path where to save failures.
    """
    with open(asin_file) as asin_file:
        json_obj = json.load(asin_file)
        asin_list = (key for key in json_obj.keys())

        file_names = [os.path.basename(file).split(".")[0] for file in metadata_files]

        with open(output_path, "w") as output_file:
            with open(failure_path, "w") as failure_file:
                for asin in asin_list:
                    found = False
                    for file_name in file_names:
                        query_result = query_by_asin(file_name, asin, cache_path)
                        if len(query_result) > 0:
                            item = query_result[0]
                            output_file.write(json.dumps(item) + "\n")
                            found = True
                            break
                    if not found:
                        failure_file.write(asin + "\n")
                failure_file.close()
            output_file.close()
