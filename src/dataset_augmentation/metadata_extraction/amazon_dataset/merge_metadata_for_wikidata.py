import orjson
import os
import re
from typing import Any, Optional, Union, Literal

from tqdm.auto import tqdm

from src.dataset_augmentation.metadata_extraction.amazon_dataset.utils import (
    correct_missing_types,
)


def _extract_year_from_title_tags(title: Optional[str]) -> Optional[int]:
    if title is None:
        return None

    year = None
    year_groups = re.search(r"\((19\d{2}|20\d{2})\)|\[(19\d{2}|20\d{2})\]", title)
    if year_groups is not None:
        year = year_groups.group(1) or year_groups.group(2)
        year = int(year)
    return year


def _extract_year_from_details(obj: dict[str, Any]) -> Optional[int]:
    if "details" not in obj:
        return None

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
        return sorted(years)[0]
    return None


def _remove_tags(title: Optional[str], mtype: str) -> Optional[str]:
    if title is None:
        return None

    # remove tags between square and round braces
    title = re.sub(r"[\[\(].*?[\]\)]", "", title)
    title = title.rstrip()

    # remove tags without braces
    if title.endswith("DVD") or title.endswith("VHS"):
        title = title[:-3].rstrip()

        # remove explicit lyrics warning
    if mtype == "music" and title.endswith("explicit_lyrics"):
        title = title[: -len("explicit_lyrics")].rstrip()

    return title.rstrip()


def _clean_title(title: Optional[str]) -> Optional[str]:
    if title is None:
        return None

    # remove common patterns
    all_pattern = re.compile(
        r"^.*?(?=\s*(?:\bthe\b\s)?[.,:-]?\s?(?:\bvolume\b|\bseason\b|\bvol\b\.?|\bcomplete\b|\bprograms\b|\bset\b|("
        r"?:\s*\b\w*[^\s\w]\w*\b|\b\w+\b\s*){0,2}\bedition\b|\bcollection\b\s*$|\bcollector("
        r"?:\'s)?\b\s\b\w*\b|\bwidescreen\b)|\s*$)",
        re.IGNORECASE,
    )
    groups = re.match(all_pattern, title)
    if groups:
        title = groups.group(0)

    # remove trailing special character and whitespace
    title = re.sub(
        r"[,.\\<>?;:\'\"\[{\]}`~!@#$%^&*()\-_=+|\s]*$",
        "",
        title,
    )
    return title.rstrip()


def _extract_person_from_details(obj: dict[str, Any], mtype: str) -> list[str]:
    person: list[str] = []
    if mtype == "movies":
        if "details" in obj and "director" in obj["details"]:
            person = obj["details"]["director"]
    elif mtype == "books":
        if "author" in obj and obj["author"] is not None:
            person = obj["author"]["name"]
    elif mtype == "music":
        if "author" in obj and obj["author"] is not None:
            person = obj["author"]["name"]
        elif "details" in obj and "contributor" in obj["details"]:
            contributor: str = obj["details"]["contributor"]
            person = [s.strip() for s in contributor.split(",")]
    else:
        raise RuntimeError("Unrecognized metadata type")
    return person


def merge_metadata_for_wikidata(
    extracted_metadata_file_path: str, merged_output_path: str
) -> None:
    base_path = os.path.join("data", "processed")
    complete_books_file = os.path.join(base_path, "complete-books.json")
    complete_music_file = os.path.join(base_path, "complete-music.json")
    complete_movies_file = os.path.join(base_path, "complete-movies.json")
    file_paths = [
        (complete_books_file, "books"),
        (complete_movies_file, "movies"),
        (complete_music_file, "music"),
    ]
    output_data = {}

    with open(extracted_metadata_file_path, "rb") as f:
        amazon_2023_extracted_metadata = orjson.loads(f.read())

    for filePath, mtype in file_paths:
        with open(filePath, "rb") as file:
            json_data = file.read()

        parsed_data = orjson.loads(json_data)

        for asin, scraped_data in tqdm(
            parsed_data.items(), desc=f"Processing {mtype} metadata", leave=False
        ):
            if isinstance(scraped_data, str):
                scraped_data = {"title": None, "person": None, "year": None}

            title = scraped_data["title"]
            person = scraped_data["person"]
            year = scraped_data["year"]

            if asin in amazon_2023_extracted_metadata:
                amazon_2023_obj: dict[str, Any] = amazon_2023_extracted_metadata[asin]
                if title is None:
                    title = amazon_2023_obj["title"]
                if person is None:
                    person = _extract_person_from_details(amazon_2023_obj, mtype)
                if year is None:
                    year = _extract_year_from_details(amazon_2023_obj)
                if year is None:
                    year = _extract_year_from_title_tags(amazon_2023_obj["title"])

            title_without_tags = _remove_tags(title, mtype)
            title_cleaned = _clean_title(title_without_tags)
            if person is None:
                person = []
            elif not isinstance(person, list):
                person = [person]

            metadata_source = {
                "title": "Amazon dataset",
                "person": ("Amazon dataset" if len(person) > 0 else None),
                "year": "Amazon dataset" if year is not None else None,
            }
            if title is not None:
                output_data[asin] = {
                    "title": title,
                    "title_without_tags": title_without_tags,
                    "title_cleaned": title_cleaned,
                    "person": person,
                    "year": year,
                    "metadata_source": metadata_source,
                    "type": mtype,
                }

    with open(merged_output_path, "wb") as output_file:
        output_file.write(orjson.dumps(output_data, option=orjson.OPT_INDENT_2))
