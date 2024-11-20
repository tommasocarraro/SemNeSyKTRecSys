import re
from typing import Any

from loguru import logger

from config import GOOGLE_API_KEY
from src.dataset_augmentation.api_querying.get_request_with_limiter import (
    get_request_with_limiter,
)
from src.dataset_augmentation.api_querying.utils import (
    get_async_limiter,
    process_http_requests,
    process_responses_with_joblib,
)


def _extract_info(res: Any):
    title, year = None, None
    try:
        for item in res["itemListElement"]:
            base_body = item["result"]
            if "TVSeries" in base_body["@type"]:
                title = base_body["name"]
                release_date = base_body["detailedDescription"]["articleBody"]
                pattern = r"\b(19|20)\d{2}\b"
                match = re.search(pattern, release_date)
                if match:
                    year = match.group(0)
                else:
                    year = None
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


async def get_shows_info(show_titles: list[str]):
    """
    Given a list of show titles, asynchronously queries the Google Graph Search API.
    Args:
        show_titles: list of show titles to be searched

    Returns: a coroutine which provides all the responses' bodies' in json format when awaited
    """
    limiter = get_async_limiter(how_many=len(show_titles), max_rate=240, time_period=60)
    tasks = [
        get_request_with_limiter(
            limiter=limiter,
            url="https://kgsearch.googleapis.com/v1/entities:search",
            title=title,
            params={
                "query": title,
                "key": GOOGLE_API_KEY,
                "limit": 10,
                "indent": "True",
                "types": ["TVSeries"],
            },
        )
        for title in show_titles
    ]

    responses = await process_http_requests(
        tasks=tasks, tqdm_desc="Querying Google KG Search API..."
    )

    return process_responses_with_joblib(responses=responses, fn=_extract_info)
