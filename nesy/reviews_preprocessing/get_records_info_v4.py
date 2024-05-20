import asyncio
from typing import Any
import re

from aiolimiter import AsyncLimiter

from config import GOOGLE_API_KEY
from .get_request import get_request
from .utils import run_with_async_limiter, process_responses_with_joblib


async def get_records_info(record_titles: list[str]):
    """
    Given a list of record titles, asynchronously queries the Google Graph Search API.
    Args:
        record_titles: list of record titles to be searched

    Returns: a coroutine which provides all the responses' bodies' in json format when awaited
    """
    limiter = AsyncLimiter(240)
    tasks = [
        run_with_async_limiter(
            limiter=limiter,
            fn=get_request,
            url="https://kgsearch.googleapis.com/v1/entities:search",
            title=title,
            params={
                "query": title,
                "key": GOOGLE_API_KEY,
                "limit": 1,
                "indent": "True",
            },
        )
        for title in record_titles
    ]

    responses = await asyncio.gather(*tasks)

    def extract_info(res: Any):
        title, artist, year = None, None, None
        try:
            for item in res["itemListElement"]:
                base_body = item["result"]
                if "MusicAlbum" in base_body["@type"]:
                    title = base_body["name"]
                    artist = base_body["description"].split("by")[1].lstrip()
                    release_date = base_body["detailedDescription"]["articleBody"]
                    pattern = r'\b(19|20)\d{2}\b'
                    match = re.search(pattern, release_date)
                    if match:
                        year = match.group(0)
                    else:
                        year = None
                    break
        except IndexError as e:
            print(f"Failed to retrieve the year: {e}")
        except KeyError as e:
            print(f"Failed to retrieve the data: {e}")
        except TypeError as e:
            print(f"Type mismatch between the expected and actual json structure: {e}")
        return title, artist, year

    return process_responses_with_joblib(responses=responses, fn=extract_info)
