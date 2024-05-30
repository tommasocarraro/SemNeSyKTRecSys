import json
import os
from typing import Any, Union
from tqdm.auto import tqdm
import regex as re

from nesy.dataset_augmentation.metadata_extraction.utils import correct_missing_types


def _extract_year(title: Union[str, None]) -> Union[int, None]:
    if title is None:
        return None

    year = None
    year_groups = re.search(r"\((19\d{2}|20\d{2})\)|\[(19\d{2}|20\d{2})\]", title)
    if year_groups is not None:
        year = year_groups.group(1) or year_groups.group(2)
        year = int(year)
    return year


def _clean_title(title: Union[str, None]) -> Union[str, None]:
    if title is None:
        return None

    # remove tags between square and round braces
    title = re.sub(r"[\[\(].*?[\]\)]", "", title)

    # remove tags without braces
    if title.endswith("DVD") or title.endswith("VHS"):
        title = title[:-3].rstrip()

    # remove explicit lyrics warning
    elif title.endswith("explicit_lyrics"):
        title = title[: -len("explicit_lyrics")].rstrip()

    # remove common patterns
    all_pattern = re.compile(
        r"^.*?(?=\s*(?:the\s)?(?:volume|season|vol\.|complete|programs|set)|\s*$)",
        re.IGNORECASE,
    )
    groups = re.match(all_pattern, title)
    if groups:
        title = groups.group(0)

    # remove trailing special character and whitespace
    title = re.sub(r"\s*[\W\D]\s*$", "", title)
    return title


def merge_metadata_for_wikidata(
    extracted_metadata: dict[str, Any], output_file_path: str
):
    with open(
        os.path.join("data", "processed", "legacy", "complete-filtered-metadata.json"),
        "r",
    ) as complete_filtered_metadata_file:
        complete_filtered_metadata = json.load(complete_filtered_metadata_file)

    output_data = {}

    for asin in tqdm(
        complete_filtered_metadata.keys(),
        desc="Merging metadata...",
        dynamic_ncols=True,
    ):
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

        year_from_title = _extract_year(title)
        title = _clean_title(title)
        output_data[asin] = {
            "title": title,
            "person": person,
            "year": year if year is not None else year_from_title,
            "type": mtype,
        }

    correct_missing_types(output_data)

    with open(output_file_path, "w") as output_file:
        json.dump(output_data, output_file, indent=4, ensure_ascii=False)
