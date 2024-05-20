import asyncio
from collections.abc import Coroutine
from typing import Any

import requests
from aiolimiter import AsyncLimiter

from config import GOOGLE_API_KEY
from .get_request import get_request
from .utils import run_with_async_limiter, process_responses_with_joblib


async def _get_book_info(title: str, language: str = "en") -> Coroutine:
    """
    Runs a GET request with the provided book title against the Google Books v1 API
    Args:
        title: book title to be searched
        language: language of the results, defaults to "en"

    Returns: a coroutine which provides the request's body in json format when awaited
    """
    base_url = "https://www.googleapis.com/books/v1/volumes"
    params = {
        "q": f'intitle:"{title}"',
        "key": GOOGLE_API_KEY,
        "langRestrict": language,
        "maxResults": 1,
        "projection": "lite",
    }
    try:
        return await get_request(base_url, params)
    except requests.RequestException as e:
        print(f"There was an error while retrieving item {title}: {e}")


async def get_books_info(book_titles: list[str]):
    """
    Given a list of book titles, asynchronously queries the Google Books v1 API
    Args:
        book_titles: list of book titles to be searched

    Returns: a coroutine which provides all the responses' bodies' in json format when awaited
    """
    limiter = AsyncLimiter(240)
    tasks = [
        run_with_async_limiter(limiter=limiter, fn=_get_book_info, title=title)
        for title in book_titles
    ]

    responses = await asyncio.gather(*tasks)

    def extract_info(res: Any):
        title, author, year = None, None, None
        try:
            base_body = res["items"][0]["volumeInfo"]
            title = base_body["title"]
            author = base_body["authors"][0]
            published_date = base_body["publishedDate"]
            year = published_date.split("-")[0]
        except (KeyError, IndexError) as e:
            print(f"Failed to retrieve data: {e}")
        return title, author, year

    return process_responses_with_joblib(responses=responses, fn=extract_info)
