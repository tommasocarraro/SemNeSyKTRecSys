import json
import os
from typing import Any, Union

import regex as re

from nesy.dataset_augmentation.metadata_extraction.utils import correct_missing_types


def _clean_title(title: Union[str, None]) -> Union[str, None]:
    if title is not None:
        # remove tags between square and round braces
        title_clean = re.sub(r"[\[\(].*?[\]\)]", "", title)
        # remove tags without braces
        if title_clean.endswith("DVD") or title_clean.endswith("VHS"):
            title_clean = title_clean[:-3].rstrip()
        # remove explicit lyrics warning
        elif title_clean.endswith("explicit_lyrics"):
            title_clean = title_clean[: -len("explicit_lyrics")].rstrip()
        # TODO optimize regexps
        # remove the complete x season
        title_clean = re.sub(
            r"\b(\s|(\s-\s)|(:\s)|(,\s))([Tt]he\s)?[Cc]omplete\s[\w\d]*\s[Ss]eason\b",
            "",
            title_clean,
        )
        # remove the complete season x
        title_clean = re.sub(
            r"\b(\s|(\s-\s)|(:\s)|(,\s))([Tt]he\s)?[Cc]omplete\s[Ss]eason\s\d*\b",
            "",
            title_clean,
        )
        # remove season x
        title_clean = re.sub(
            r"\b(\s|(\s-\s)|(:\s)|(,\s))[Ss]eason\s\d*\b", "", title_clean
        )
        # remove seasons x-y
        title_clean = re.sub(
            r"\b(\s|(\s-\s)|(:\s)|(,\s))[Ss]easons\s\d*-\d*\b", "", title_clean
        )
        # remove vol. x
        title_clean = re.sub(
            r"\b(\s|(\s-\s)|(:\s)|(,\s))[Vv]ol.\s\d*\b", "", title_clean
        )
        # remove volume x
        title_clean = re.sub(
            r"\b(\s|(\s-\s)|(:\s)|(,\s))[Vv]olume\s\d*\b", "", title_clean
        )
        # remove volume(s) x&y
        title_clean = re.sub(
            r"\b(\s|(\s-\s)|(:\s)|(,\s))[Vv]olumes?\s\d*\s?&\s?\d*\b", "", title_clean
        )
        # remove the complete series
        title_clean = re.sub(
            r"\b(\s|(\s-\s)|(:\s)|(,\s))[Tt]he\s[Cc]omplete\s[Ss]eries\b",
            "",
            title_clean,
        )
        # remove complete set
        title_clean = re.sub(
            r"\b(\s|(\s-\s)|(:\s)|(,\s))[Cc]omplete\s[Ss]et\b", "", title_clean
        )
        # remove programs x-y
        title_clean = re.sub(
            r"\b(\s|(\s-\s)|(:\s)|(,\s))[Pp]rograms?\s\d*-\d*\b", "", title_clean
        )
        # remove set x
        title_clean = re.sub(
            r"\b(\s|(\s-\s)|(:\s)|(,\s))[Ss]et\s\d*\b", "", title_clean
        )
        # remove complete
        title_clean = re.sub(
            r"\b(\s|(\s-\s)|(:\s)|(,\s))[Cc]omplete\b", "", title_clean
        )
        # remove trailing special character
        title_clean = re.sub(r"[^\w\s]$", "", title_clean)
        # remove any remaining whitespace
        return title_clean.rstrip()


def merge_metadata_for_wikidata(
    extracted_metadata: dict[str, Any], output_file_path: str
):
    with open(
        os.path.join("data", "processed", "legacy", "complete-filtered-metadata.json"),
        "r",
    ) as complete_filtered_metadata_file:
        complete_filtered_metadata = json.load(complete_filtered_metadata_file)

    output_data = {}

    for asin in complete_filtered_metadata.keys():
        title, person, year, mtype = None, None, None, None

        # set title initially as fallback in case it's found in complete filtered metadata
        if complete_filtered_metadata[asin] != "404-error":
            title = complete_filtered_metadata[asin]

        # check if ASIN is in extracted metadata
        if asin in extracted_metadata:
            obj = extracted_metadata[asin]
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
                        person = obj["author"]["name"]
                    elif "details" in obj and "contributor" in obj["details"]:
                        contributor: str = obj["details"]["contributor"]
                        person = [s.strip() for s in contributor.split(",")]
                    mtype = "cds_and_vinyl"
                else:
                    raise RuntimeError("Unrecognized metadata type")

        title = _clean_title(title)
        output_data[asin] = {
            "title": title,
            "person": person,
            "year": year,
            "type": mtype,
        }

    correct_missing_types(output_data)

    with open(output_file_path, "w") as output_file:
        json.dump(output_data, output_file, indent=4, ensure_ascii=False)
