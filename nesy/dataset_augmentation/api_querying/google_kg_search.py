import re
from typing import Any

from loguru import logger

from config import GOOGLE_API_KEY
from .get_request import get_request
from .utils import (
    run_with_async_limiter,
    process_responses_with_joblib,
    get_async_limiter,
    tqdm_run_tasks_async,
)


def extract_info_music(res: Any):
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


def extract_info_movie_tvseries(res: Any):
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


async def google_kg_search(
    titles: list[str],
    query_types: list[str],
    language: str = "en",
):
    """
    Given a list of titles, asynchronously queries the Google Graph Search API.
    Args:
        titles: list of titles to be searched
        query_types: types of entities to be searched
        language: languages of the results, defaults to "en"

    Returns:
        a coroutine which provides all the responses\' bodies\' in json format when awaited
    """
    max_rate = 10
    time_period = 1.0
    limiter = get_async_limiter(
        how_many=len(titles), max_rate=max_rate, time_period=time_period
    )
    if "Book" in query_types:
        extract_info_cb = None
    elif "CDs" in query_types:
        extract_info_cb = None
    else:
        extract_info_cb = extract_info_movie_tvseries
    tasks = [
        run_with_async_limiter(
            limiter=limiter,
            fn=get_request,
            url="https://kgsearch.googleapis.com/v1/entities:search",
            title=title,
            params=[
                ("query", title),
                ("key", GOOGLE_API_KEY),
                ("limit", 10),
                ("indent", "True"),
                ("languages", language),
                *list(zip(["types" for _ in range(len(query_types))], query_types)),
            ],
        )
        for title in titles
    ]

    responses = await tqdm_run_tasks_async(tasks)

    return process_responses_with_joblib(responses=responses, fn=extract_info_cb)
