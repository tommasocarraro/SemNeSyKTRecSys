import asyncio
from typing import Any, Union

from aiolimiter import AsyncLimiter
from loguru import logger

from config import LAST_FM_API_KEY
from .get_request import get_request
from .utils import run_with_async_limiter, process_responses_with_joblib


async def get_records_info(records_titles: list[str]):
    """
    Given a list of book titles, asynchronously queries the Google Books v1 API
    Args:
        records_titles: list of record titles to be searched

    Returns: a coroutine which provides all the responses\' bodies\' in json format when awaited
    """
    limiter = AsyncLimiter(5)
    tasks = [
        run_with_async_limiter(
            limiter=limiter,
            fn=get_request,
            title=title,
            url="https://ws.audioscrobbler.com/2.0",
            params={
                "album": title,
                "api_key": LAST_FM_API_KEY,
                "method": "album.search",
                "format": "json",
                "limit": 1,
            },
        )
        for title in records_titles
    ]
    search_results = await asyncio.gather(*tasks)

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

    search_info = process_responses_with_joblib(
        responses=search_results, fn=_extract_search_info
    )

    tasks = [
        run_with_async_limiter(
            limiter=limiter,
            fn=get_request,
            url="https://ws.audioscrobbler.com/2.0",
            params={
                "artist": artist,
                "album": album,
                "api_key": LAST_FM_API_KEY,
                "method": "album.getinfo",
                "format": "json",
                "limit": 1,
            },
            title=artist,
        )
        for album, artist in search_info
    ]
    info = await asyncio.gather(*tasks)

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

    return process_responses_with_joblib(responses=info, fn=_extract_record_info)
