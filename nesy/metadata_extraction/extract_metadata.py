import json
import os.path

from .query_by_asin import query_by_asin


def extract_metadata(
    asin_file_path: str, metadata_files: list[str], cache_path: str
) -> None:
    """
    For each ASIN code found in asin_file, extract the respective metadata from the SQLITE database.
    Args:
        asin_file_path: File containing the ASIN codes.
        metadata_files: List of metadata files.
        cache_path: Path to the SQLITE database.
    """
    output_dir = os.path.dirname(asin_file_path)

    # read all ASIN codes from the ASIN file
    with open(asin_file_path) as asin_file:
        json_obj = json.load(asin_file)
        asin_list = (key for key in json_obj.keys())

    # extract the input metadata file names without extensions
    file_names = [os.path.basename(file).split(".")[0] for file in metadata_files]

    # open all output files and keep track of the file variables in a dictionary
    output_files = {
        filename: open(
            os.path.join(output_dir, "extracted_metadata_" + filename + ".jsonl"), "w"
        )
        for filename in file_names
    }

    # store each entry both in a common file and in its own category's file
    with open(os.path.join(output_dir, "extracted_metadata.jsonl"), "w") as output_file:
        with open(os.path.join(output_dir, "failed_metadata.txt"), "w") as failure_file:
            for asin in asin_list:
                found = False
                for file_name in file_names:
                    if not found:
                        query_result = query_by_asin(file_name, asin, cache_path)
                        if len(query_result) > 0:
                            item = query_result[0]
                            output_file.write(json.dumps(item) + "\n")
                            output_files[file_name].write(json.dumps(item) + "\n")
                            found = True
                if not found:
                    failure_file.write(asin + "\n")

    for file in output_files.values():
        file.close()
