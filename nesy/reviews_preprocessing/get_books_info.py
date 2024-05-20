import asyncio
from typing import Any

from aiolimiter import AsyncLimiter

from config import GOOGLE_API_KEY
from .get_request import get_request
from .utils import run_with_async_limiter, process_responses_with_joblib


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
            limiter=limiter,
            fn=get_request,
            url="https://www.googleapis.com/books/v1/volumes",
            title=title,
            params={
                "q": f'intitle:"{title}"',
                "key": GOOGLE_API_KEY,
                "langRestrict": "en",
                "maxResults": 1,
                "projection": "lite",
            },
        )
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
