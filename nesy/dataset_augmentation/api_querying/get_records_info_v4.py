import asyncio
import re
from typing import Any

from aiolimiter import AsyncLimiter
from loguru import logger

from config import GOOGLE_API_KEY
from .get_request import get_request_with_limiter
from .utils import (
    process_http_requests,
    process_responses_with_joblib,
    get_async_limiter,
)


def _extract_info(res: Any):
    title, artist, year = None, None, None
    try:
        for item in res["itemListElement"]:
            base_body = item["result"]
            if "MusicAlbum" in base_body["@type"]:
                title = base_body["name"]
                artist = base_body["description"].split("by")[1].lstrip()
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
    return title, artist, year


async def get_records_info(record_titles: list[str]):
    """
    Given a list of record titles, asynchronously queries the Google Graph Search API.
    Args:
        record_titles: list of record titles to be searched

    Returns: a coroutine which provides all the responses' bodies' in json format when awaited
    """
    limiter = get_async_limiter(how_many=len(record_titles), max_rate=10, time_period=1)
    tasks = [
        get_request_with_limiter(
            url="https://kgsearch.googleapis.com/v1/entities:search",
            title=title,
            params={
                "query": title,
                "key": GOOGLE_API_KEY,
                "limit": 1,
                "indent": "True",
            },
            limiter=limiter,
        )
        for title in record_titles
    ]
    responses = await process_http_requests(
        tasks=tasks, tqdm_desc="Querying Google KG Search API..."
    )

    return process_responses_with_joblib(responses=responses, fn=_extract_info)
