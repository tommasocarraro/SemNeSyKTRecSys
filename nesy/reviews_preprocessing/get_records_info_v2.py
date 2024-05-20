import asyncio
from typing import Any, Optional

from aiolimiter import AsyncLimiter

from .get_request import get_request
from .utils import run_with_async_limiter, process_responses_with_joblib


async def get_records_info(record_titles: list[str]):
    """
    Given a list of record titles, asynchronously queries the MusicBrainz v2 API
    Args:
        record_titles: list of record titles to be searched

    Returns: a coroutine which provides all the responses' bodies' in json format when awaited
    """
    limiter = AsyncLimiter(max_rate=1, time_period=1)
    tasks = [
        run_with_async_limiter(
            limiter=limiter,
            fn=get_request,
            title=title,
            url="https://musicbrainz.org/ws/2/release",
            params={"query": title, "limit": 1, "fmt": "json"},
        )
        for title in record_titles
    ]

    def extract_info(res: Any) -> tuple[Optional[str], Optional[str], Optional[str]]:
        title, artist, year = None, None, None
        try:
            base_body = res["releases"][0]
            title = base_body["title"]
            artist = base_body["artist-credit"][0]["name"]
            release_date = base_body["date"]
            year = release_date.split("-")[0]
        except (IndexError, KeyError) as e:
            print(f"Failed to retrieve the information: {e}")
        return title, artist, year

    responses = await asyncio.gather(*tasks)

    filtered_artists = [
        artist for artist in responses[0]["releases"] if "disambiguation" not in artist
    ]

    for artist in filtered_artists:
        print(artist)

    return process_responses_with_joblib(responses=responses, fn=extract_info)
