from asyncio.exceptions import CancelledError
from json import JSONDecodeError
from typing import Union

import aiohttp
from aiohttp import ClientResponseError
from aiolimiter import AsyncLimiter
from backoff import expo, on_exception, random_jitter
from loguru import logger


async def _handle_backoff(details):
    limiter: AsyncLimiter = details["kwargs"]["_limiter"]
    logger.warning(
        f"HTTP error 429 received. Setting limiter time period to {limiter.time_period} seconds"
    )
    limiter.time_period *= 2


@on_exception(
    expo,
    ClientResponseError,
    max_tries=8,
    on_backoff=_handle_backoff,
    jitter=random_jitter,
    giveup=lambda e: e.status not in [429, 503],
)
async def _fetch(
    url: str,
    params: Union[dict[str, str], list[tuple[str, str]]],
    _limiter: AsyncLimiter,
):
    headers = {
        "accept": "application/json",
        "User-Agent": "SemNeSyKTRecSys-Python ( nicolo.bertocco@studenti.unipd.it )",
        "Accept-Charset": "utf-8",
    }
    async with aiohttp.ClientSession(raise_for_status=True) as session:
        async with session.get(url, params=params, headers=headers) as response:
            return await response.json()


async def get_request_with_limiter(
    url: str,
    title: str,
    params: Union[dict[str, str], list[tuple[str, str]]],
    limiter: AsyncLimiter,
):
    """
    Runs a GET request against the provided URL's API with the provided params
    Args:
        url: target URL
        title: title to be searched, only used for debugging purposes
        params: dictionary of HTTP parameters
        limiter: an aiolimiter AsyncLimiter

    Returns:
        a coroutine which provides the request\'s body in json format when awaited
    """
    try:
        async with limiter:
            body = await _fetch(url=url, params=params, _limiter=limiter)
            return title, body
    except (CancelledError, ClientResponseError, JSONDecodeError) as e:
        raise e
    finally:
        pass
