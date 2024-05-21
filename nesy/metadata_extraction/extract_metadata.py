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

    # dictionary containing all opened output files and a dictionary which will be filled with the data to write in them
    output_data_and_files = {
        filename: (
            open(
                os.path.join(output_dir, "extracted_metadata_" + filename + ".json"),
                "w",
            ),
            {},
        )
        for filename in file_names + ["all", "misses"]
    }

    for asin in asin_list:
        found = False
        for file_name in file_names:
            if not found:
                query_result = query_by_asin(file_name, asin, cache_path)
                if len(query_result) > 0:
                    item = query_result[0]
                    asin = item["parent_asin"]

                    lower_item = {
                        k.lower(): v for k, v in item.items() if k != "parent_asin"
                    }

                    output_data_and_files["all"][1][asin] = lower_item
                    # dicts are actually passed by reference so this adds the type field to all files, deepcopy would
                    # be expensive
                    output_data_and_files["all"][1][asin]["type"] = file_name.lower()
                    output_data_and_files[file_name][1][asin] = lower_item
                    found = True
        if not found:
            output_data_and_files["all"][1][asin] = {}
            output_data_and_files["misses"][1][asin] = {}

    # write JSON objects to file system and close file connections
    for file, data in output_data_and_files.values():
        file.write(json.dumps(data))
        file.close()
