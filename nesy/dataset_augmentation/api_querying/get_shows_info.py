import asyncio
from typing import Any

from aiolimiter import AsyncLimiter

from config import TMDB_API_KEY
from .get_request import get_request
from .utils import run_with_async_limiter, process_responses_with_joblib


async def get_shows_info(show_titles: list[str]):
    """
    Given a list of show titles, asynchronously queries the TMDB API
    Args:
        show_titles: list of show titles to be searched

    Returns: a coroutine which provides all the responses\' bodies\' in json format when awaited
    """
    limiter = AsyncLimiter(45, time_period=1)
    tasks = [
        run_with_async_limiter(
            limiter=limiter,
            fn=get_request,
            title=title,
            url="https://api.themoviedb.org/3/search/tv",
            params={
                "query": title,
                "api_key": TMDB_API_KEY,
                "include_adult": "false",
                "page": 1,
            },
        )
        for title in show_titles
    ]

    def extract_info(res: Any):
        title, year = None, None
        try:
            base_body = res["results"][0]
            title = base_body["original_name"]
            release_date = base_body["first_air_date"]
            year = release_date.split(sep="-")[0]
        except IndexError as e:
            print(f"Failed to retrieve the year: {e}")
        except KeyError as e:
            print(f"Failed to retrieve the data: {e}")
        except TypeError as e:
            print(f"Type mismatch between the expected and actual json structure: {e}")
        return title, year

    responses = await asyncio.gather(*tasks)
    return process_responses_with_joblib(extract_info, responses)
