import os.path

import orjson
from tqdm.auto import tqdm

from .query_by_asin import query_by_asin


def extract_metadata(
    asin_file_path: str, metadata_files: list[str], cache_path: str, output_path: str
):
    """
    For each ASIN code found in asin_file, extract the respective metadata from the SQLITE database.
    Args:
        asin_file_path: File containing the ASIN codes.
        metadata_files: List of metadata files.
        cache_path: Path to the SQLITE database.
        output_path: Output path.
    """
    # read all ASIN codes from the ASIN file
    with open(asin_file_path) as asin_file:
        json_obj = orjson.loads(asin_file.read())
        asin_list = [key for key in json_obj.keys()]

    # extract the input metadata file names without extensions
    file_names = [os.path.basename(file).split(".")[0] for file in metadata_files]
    output_data = {}
    for asin in tqdm(
        asin_list, desc="Extracting metadata...", dynamic_ncols=True, leave=False
    ):
        found = False
        for file_name in file_names:
            if not found:
                query_result = query_by_asin(file_name, asin, cache_path)
                if len(query_result) > 0:
                    item = query_result[0]
                    asin = item["parent_asin"]

                    def lowercase_keys(d):
                        if isinstance(d, dict):
                            return {k.lower(): lowercase_keys(v) for k, v in d.items()}
                        elif isinstance(d, list):
                            return [lowercase_keys(item) for item in d]
                        else:
                            return d

                    lower_item = lowercase_keys(item)
                    if "books" in file_name.lower():
                        lower_item["type"] = "books"
                    elif "movies" in file_name.lower():
                        lower_item["type"] = "movies"
                    elif "cds" in file_name.lower():
                        lower_item["type"] = "music"

                    output_data[asin] = lower_item
                    found = True

    with open(output_path, mode="wb") as output_file:
        output_file.write(orjson.dumps(output_data, option=orjson.OPT_INDENT_2))
