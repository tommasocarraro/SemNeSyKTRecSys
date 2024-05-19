import asyncio
from collections.abc import Coroutine

from aiolimiter import AsyncLimiter

from config import GOOGLE_API_KEY
from .get_request import get_request
from ..utils import run_with_async_limiter


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
        "q": title,
        "key": GOOGLE_API_KEY,
        "langRestrict": language,
        "maxResults": 1,
    }
    return await get_request(base_url, params)


async def get_books_info(book_titles: list[str]):
    """
    Given a list of book titles, asynchronously queries the Google Books v1 API
    Args:
        book_titles: list of book titles to be searched

    Returns: a coroutine which provides all the responses' bodies' in json format when awaited
    """
    limiter = AsyncLimiter(240)
    tasks = [
        run_with_async_limiter(
            limiter=limiter, fn=_get_book_info, params={title: title}
        )
        for title in book_titles
    ]
    return await asyncio.gather(*tasks)
