from typing import Any

from loguru import logger

from config import GOOGLE_API_KEY
from src.dataset_augmentation.api_querying.get_request_with_limiter import (
    get_request_with_limiter,
)
from src.dataset_augmentation.api_querying.utils import (
    process_responses_with_joblib,
    get_async_limiter,
    process_http_requests,
)


def _extract_info(res: Any):
    title, year = None, None
    try:
        for item in res["itemListElement"]:
            base_body = item["result"]
            if "Movie" in base_body["@type"]:
                title = base_body["name"]
                release_date = base_body["description"]
                year = release_date.split()[0]
                break
    except IndexError as e:
        logger.error(f"Failed to retrieve the year: {e}")
    except KeyError as e:
        logger.error(f"Failed to retrieve the data: {e}")
    except TypeError as e:
        logger.error(
            f"Type mismatch between the expected and actual json structure: {e}"
        )
    return title, year


async def get_movies_info(movie_titles: list[str]):
    """
    Given a list of movie titles, asynchronously queries the Google Graph Search API.
    Args:
        movie_titles: list of movie titles to be searched

    Returns: a coroutine which provides all the responses' bodies' in json format when awaited
    """
    limiter = get_async_limiter(
        how_many=len(movie_titles), max_rate=240, time_period=60
    )

    tasks = [
        get_request_with_limiter(
            url="https://kgsearch.googleapis.com/v1/entities:search",
            title=title,
            params={
                "query": title,
                "key": GOOGLE_API_KEY,
                "limit": 10,
                "indent": "True",
                "types": ["Movie"],
            },
            limiter=limiter,
        )
        for title in movie_titles
    ]

    responses = await process_http_requests(
        tasks=tasks, tqdm_desc="Querying Google KG Search..."
    )

    return process_responses_with_joblib(responses=responses, fn=_extract_info)
