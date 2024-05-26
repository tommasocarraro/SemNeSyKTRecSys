import asyncio
from typing import Any, Optional

from loguru import logger

from .get_request import get_request
from .utils import (
    run_with_async_limiter,
    process_responses_with_joblib,
    get_async_limiter,
    tqdm_run_tasks_async,
)


async def get_records_info(record_titles: list[str]):
    """
    Given a list of record titles, asynchronously queries the MusicBrainz v2 API
    Args:
        record_titles: list of record titles to be searched

    Returns:
        a coroutine which provides all the responses' bodies' in json format when awaited
    """
    limiter = get_async_limiter(how_many=len(record_titles), max_rate=1, time_period=1)
    tasks = [
        run_with_async_limiter(
            limiter=limiter,
            fn=get_request,
            title=title,
            url="https://musicbrainz.org/ws/2/release",
            params={"query": title, "limit": 1, "fmt": "json"},
        )
        for title in record_titles
    ]
    responses = await tqdm_run_tasks_async(tasks, desc="Querying MusicBrainz...")

    def extract_info(
        data: tuple[str, dict[str, Any]]
    ) -> tuple[str, Optional[str], Optional[str]]:
        artist, year = None, None
        title, res = data
        try:
            base_body = res["releases"][0]
            try:
                artist = base_body["artist-credit"][0]["name"]
            except (KeyError, IndexError):
                logger.warning(f"Failed to retrieve {title}'s artist")
            try:
                release_date = base_body["date"]
                year = release_date.split("-")[0]
            except KeyError:
                logger.warning(f"Failed to retrieve {title}'s release year")
            except IndexError:
                logger.warning(
                    f"Something went wrong when extracting the year from the release date"
                )
        except KeyError as e:
            logger.error(f"Failed to retrieve the information: {e}")
        except IndexError:
            logger.warning(f"No releases found for {title}")
        except TypeError as e:
            logger.error(
                f"Something went wrong while retrieving {title}'s information: {e}"
            )
        return title, artist, year

    music_info = process_responses_with_joblib(responses=responses, fn=extract_info)
    music_dict = {
        title: {"title": title, "person": artist, "year": year}
        for title, artist, year in music_info
    }
    return music_dict
