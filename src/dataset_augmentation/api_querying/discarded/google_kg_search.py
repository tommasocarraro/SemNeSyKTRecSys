from typing import Any

import regex as re
from loguru import logger


def _extract_info_music(res: Any):
    artist, year = None, None
    try:
        for item in res["itemListElement"]:
            base_body = item["result"]
            artist = base_body["description"].split("by")[1].strip()
            release_date = base_body["detailedDescription"]["articleBody"]
            pattern = r"\b(19|20)\d{2}\b"
            match = re.search(pattern, release_date)
            if match:
                year = match.group(0)
            break
    except IndexError as e:
        logger.error(f"Failed to retrieve the artist: {e}")
    except KeyError as e:
        logger.error(f"Failed to retrieve the data: {e}")
    except TypeError as e:
        logger.error(
            f"Type mismatch between the expected and actual json structure: {e}"
        )
    return artist, year


def _extract_info_movie_tvseries(res: Any):
    director, year = None, None
    for item in res[1]["itemListElement"]:
        base_body = item["result"]
        detailed_description = base_body["detailedDescription"]["articleBody"]
        pattern = r"\b(19|20)\d{2}\b"
        match = re.search(pattern, detailed_description)
        if match:
            year = match.group(0)
        pattern = r"(?i)directed by ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
        match = re.search(pattern, detailed_description)
        if match:
            director = match.group(0).lstrip("directed by ")
        else:
            pattern = r"(?i)by ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
            match = re.search(pattern, detailed_description)
            if match:
                director = match.group(0).lstrip("by ")
        if director is not None and year is not None:
            return director, year
    return director, year
