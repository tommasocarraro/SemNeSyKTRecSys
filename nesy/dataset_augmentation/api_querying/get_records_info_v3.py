from typing import Any

from loguru import logger

from config import LAST_FM_API_KEY
from .get_request import get_request_with_limiter
from .utils import (
    process_responses_with_joblib,
    get_async_limiter,
    process_http_requests,
)


def _extract_info(json_response: Any):
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


async def get_records_info(records_titles: list[str]):
    """
    Given a list of book titles, asynchronously queries the Google Books v1 API
    Args:
        records_titles: list of record titles to be searched

    Returns: a coroutine which provides all the responses\' bodies\' in json format when awaited
    """
    limiter = get_async_limiter(
        how_many=len(records_titles), max_rate=5, time_period=60
    )
    tasks = [
        get_request_with_limiter(
            url="https://ws.audioscrobbler.com/2.0",
            title=title,
            params={
                "album": title,
                "api_key": LAST_FM_API_KEY,
                "method": "album.search",
                "format": "json",
                "limit": 1,
            },
            limiter=limiter,
        )
        for title in records_titles
    ]
    responses = await process_http_requests(
        tasks=tasks, tqdm_desc="Querying last.fm..."
    )

    return process_responses_with_joblib(responses=responses, fn=_extract_info)
