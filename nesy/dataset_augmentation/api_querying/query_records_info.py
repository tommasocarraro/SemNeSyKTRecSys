from typing import Any, Union

from loguru import logger

from .get_request_with_limiter import get_request_with_limiter
from .utils import (
    process_responses_with_joblib,
    get_async_limiter,
    process_http_requests,
)


def _extract_info(data: Union[tuple[str, dict[str, Any]], tuple[str, None]]):
    artist, year = None, None
    title, res = data
    err = False
    if res is None:
        err = True
    else:
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
            err = True
    return title, artist, year, err


async def get_records_info(record_titles: list[str]):
    """
    Given a list of record titles, asynchronously queries the MusicBrainz v2 API
    Args:
        record_titles: list of record titles to be searched

    Returns:
        a coroutine which provides all the responses' bodies\' in json format when awaited
    """
    limiter = get_async_limiter(how_many=len(record_titles), max_rate=1, time_period=1)
    tasks = [
        get_request_with_limiter(
            url="https://musicbrainz.org/ws/2/release",
            title=title,
            params={"query": title, "limit": 1, "fmt": "json"},
            limiter=limiter,
        )
        for title in record_titles
    ]
    responses = await process_http_requests(
        tasks=tasks, tqdm_desc="Querying MusicBrainz..."
    )

    music_info = process_responses_with_joblib(responses=responses, fn=_extract_info)
    return {
        title: {"title": title, "person": artist, "year": year, "err": err}
        for title, artist, year, err in music_info
    }
