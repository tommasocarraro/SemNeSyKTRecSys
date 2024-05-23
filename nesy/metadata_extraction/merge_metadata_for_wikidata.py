import json
import os

import regex as re

from .utils import correct_missing_types


def merge_metadata_for_wikidata(output_file_path: str):
    extracted_metadata_all_file = open(
        os.path.join("data", "processed", "extracted_metadata_all.json"), "r"
    )
    extracted_metadata_all = json.load(extracted_metadata_all_file)

    complete_filtered_metadata_file = open(
        os.path.join("data", "processed", "legacy", "complete-filtered-metadata.json"),
        "r",
    )
    complete_filtered_metadata = json.load(complete_filtered_metadata_file)

    output_file = open(output_file_path, "w")
    output_data = {}

    for asin in complete_filtered_metadata.keys():
        title, person, year, mtype = None, None, None, None

        # set title initially as fallback in case it's found in complete filtered metadata
        if complete_filtered_metadata[asin] != "404-error":
            title = complete_filtered_metadata[asin]

        # check if ASIN is in extracted metadata
        if asin in extracted_metadata_all:
            obj = extracted_metadata_all[asin]
            # check if object is not empty
            if bool(obj):
                # try to extract the title
                if "title" in obj and obj["title"] is not None:
                    title = obj["title"]

                # try to extract the publication year
                if "details" in obj:
                    all_keys: list[str] = obj["details"].keys()
                    dates = []
                    for key in all_keys:
                        if "date" in key.lower():
                            dates.append(key)

                    years = []
                    reg = r"\b\d{4}\b"
                    for date in dates:
                        reg_matches = re.findall(reg, obj["details"][date])
                        if len(reg_matches) > 0:
                            years.append(int(sorted(reg_matches)[0]))
                    if len(years) > 0:
                        year = sorted(years)[0]

                # extracting person requires accessing specific fields depending on the type
                if "movies" in obj["type"]:
                    if "details" in obj and "director" in obj["details"]:
                        person = obj["details"]["director"]
                    mtype = "movies_and_tv"
                elif "books" in obj["type"]:
                    if "author" in obj and obj["author"] is not None:
                        person = obj["author"]["name"]
                    mtype = "books"
                elif "cds" in obj["type"]:
                    if "author" in obj and obj["author"] is not None:
                        person = obj["author"]
                    elif "details" in obj and "contributor" in obj["details"]:
                        contributor: str = obj["details"]["contributor"]
                        contributors = [s.strip() for s in contributor.split(",")]
                        if len(contributors) == 1:
                            person = contributors[0]
                    mtype = "cds_and_vinyl"
                else:
                    raise RuntimeError("Unrecognized metadata type")
        output_data[asin] = {
            "title": title,
            "person": person,
            "year": year,
            "type": mtype,
        }

    correct_missing_types(output_data)

    json.dump(output_data, output_file, indent=4, ensure_ascii=False)
    output_file.close()
    complete_filtered_metadata_file.close()
    extracted_metadata_all_file.close()
