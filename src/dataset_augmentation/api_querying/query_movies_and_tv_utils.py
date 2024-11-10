from typing import Any, Optional


def extract_movie_director(item: dict[str, Any]) -> Optional[str]:
    try:
        crew = item["crew"]
        for member in crew:
            if member["job"] == "Director":
                return member["name"]
    except (KeyError, TypeError, IndexError):
        return None


def extract_movie_year(item: dict[str, Any]) -> Optional[str]:
    if "release_date" in item:
        release_date = item["release_date"]
        return release_date.split("-")[0]


def extract_movie_id(item: dict[str, Any]) -> Optional[str]:
    if "id" in item:
        return item["id"]


def extract_tv_year(item: dict[str, Any]) -> Optional[str]:
    if "first_air_date" in item:
        first_air_date = item["first_air_date"]
        return first_air_date.split("-")[0]


def extract_title(item: dict[str, Any]) -> str:
    if item["media_type"] == "tv":
        if "original_name" in item:
            return item["original_name"]
        elif "name" in item:
            return item["name"]
        else:
            raise ValueError(f"Title not found in specified fields: {item}")
    elif item["media_type"] == "movie":
        if "original_title" in item:
            return item["original_title"]
        elif "title" in item:
            return item["title"]
        else:
            raise ValueError(f"Title not found in specified fields: {item}")
    else:
        raise ValueError(f"Unexpected media type: {item['media_type']}")
