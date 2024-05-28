from typing import Any, Union

from loguru import logger

from config import LAST_FM_API_KEY
from nesy.dataset_augmentation.api_querying.get_request_with_limiter import (
    get_request_with_limiter,
)
from nesy.dataset_augmentation.api_querying.utils import (
    process_http_requests,
    process_responses_with_joblib,
    get_async_limiter,
)


def _extract_search_info(json_response: Any):
    try:
        base_body = json_response["results"]["albummatches"]["album"][0]
        artist = base_body["artist"]
        name = base_body["name"]
        return name, artist
    except KeyError as e:
        logger.error(f"There was an error while exploring the json structure: {e}")
        return None, None
    except IndexError as e:
        logger.error(f"The results are empty: {e}")
        return None, None


def _extract_record_info(json_response: Any):
    name: Union[str, None] = None
    artist: Union[str, None] = None
    try:
        base_body = json_response["album"]
        artist = base_body["artist"]
        name = base_body["name"]
        published: str = base_body["wiki"]["published"]
        year = published.split(sep=" ")[2][:-1]
        return name, artist, year
    except KeyError as e:
        logger.error(f"There was an error while exploring the json structure: {e}")
        return None, None, None
    except IndexError as e:
        logger.error(f"Couldn't retrieve the date: {e}")
        return name, artist, None


async def get_records_info(records_titles: list[str]):
    """
    Given a list of book titles, asynchronously queries the Google Books v1 API
    Args:
        records_titles: list of record titles to be searched

    Returns: a coroutine which provides all the responses\' bodies\' in json format when awaited
    """
    limiter = get_async_limiter(how_many=len(records_titles), max_rate=5, time_period=1)
    tasks = [
        get_request_with_limiter(
            url="https://ws.audioscrobbler.com/2.0",
            params={
                "album": title,
                "api_key": LAST_FM_API_KEY,
                "method": "album.search",
                "format": "json",
                "limit": 1,
            },
            title=title,
            limiter=limiter,
        )
        for title in records_titles
    ]
    search_results = await process_http_requests(
        tasks=tasks, tqdm_desc="Querying last.fm..."
    )

    search_info = process_responses_with_joblib(
        responses=search_results, fn=_extract_search_info
    )

    tasks = [
        get_request_with_limiter(
            url="https://ws.audioscrobbler.com/2.0",
            params={
                "artist": artist,
                "album": album,
                "api_key": LAST_FM_API_KEY,
                "method": "album.getinfo",
                "format": "json",
                "limit": 1,
            },
            title=album,
            limiter=limiter,
        )
        for album, artist in search_info
    ]
    info = await process_http_requests(tasks=tasks, tqdm_desc="Querying last.fm...")

    return process_responses_with_joblib(responses=info, fn=_extract_record_info)
