import asyncio
from collections.abc import Coroutine
from typing import Any

from aiolimiter import AsyncLimiter

from config import TMDB_API_KEY
from .get_request import get_request
from .utils import run_with_async_limiter, process_responses_with_joblib


async def _get_movie_info(title: str) -> Coroutine:
    """
    Runs a GET request with the provided movie title against the TMDB API
    Args:
        title: movie title to be searched

    Returns: a coroutine which provides the request's body in json format when awaited
    """
    base_url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "query": title,
        "api_key": TMDB_API_KEY,
        "include_adult": "false",
        "page": 1,
    }
    return await get_request(base_url, params)


async def get_movies_info(movie_titles: list[str]):
    """
    Given a list of book titles, asynchronously queries the TMDB API
    Args:
        movie_titles: list of movie titles to be searched

    Returns: a coroutine which provides all the responses\' bodies\' in json format when awaited
    """
    limiter = AsyncLimiter(45, time_period=1)
    tasks = [
        run_with_async_limiter(limiter=limiter, fn=_get_movie_info, title=title)
        for title in movie_titles
    ]

    def extract_info(res: Any):
        title, year = None, None
        try:
            base_body = res["results"][0]
            title = base_body["original_title"]
            release_date = base_body["release_date"]
            year = release_date.split(sep="-")[0]
        except IndexError as e:
            print(f"Failed to retrieve the year: {e}")
        except KeyError as e:
            print(f"Failed to retrieve the data: {e}")
        except TypeError as e:
            print(f"Type mismatch between the expected and actual json structure: {e}")
        return title, year

    responses = await asyncio.gather(*tasks)

    return process_responses_with_joblib(responses=responses, fn=extract_info)
