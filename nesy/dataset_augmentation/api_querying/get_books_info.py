from typing import Any

from config import GOOGLE_API_KEY
from .get_request import get_request
from .utils import (
    run_with_async_limiter,
    process_responses_with_joblib,
    get_async_limiter,
    tqdm_run_tasks_async,
)


async def get_books_info(book_titles: list[str]):
    """
    Given a list of book titles, asynchronously queries the Google Books v1 API
    Args:
        book_titles: list of book titles to be searched
    Returns:
        a coroutine which provides all the responses\' bodies\' in json format when awaited
    """
    max_rate = 1
    time_period = 2.0
    limiter = get_async_limiter(
        how_many=len(book_titles), max_rate=max_rate, time_period=time_period
    )
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

    responses = await tqdm_run_tasks_async(tasks)

    def extract_info(res: tuple[str, Any]) -> tuple[str, str, str]:
        title, res_body = res
        author, year = None, None
        try:
            base_body = res_body["items"][0]["volumeInfo"]
            author = base_body["authors"][0]
            published_date = base_body["publishedDate"]
            year = published_date.split("-")[0]
        except (KeyError, IndexError) as e:
            # print(f"Failed to retrieve data: {e}")
            pass
        return title, author, year

    return process_responses_with_joblib(responses=responses, fn=extract_info)