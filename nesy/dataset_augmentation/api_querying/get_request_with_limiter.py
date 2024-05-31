from asyncio import Lock
from asyncio.exceptions import CancelledError
from datetime import datetime
from json import JSONDecodeError
from typing import Union, Any

import aiohttp
from aiohttp import ClientResponseError
from aiolimiter import AsyncLimiter
from backoff import expo, on_exception, random_jitter
from loguru import logger

last_time_backoff: Union[datetime, None] = None
last_time_success: Union[datetime, None] = None
lock = Lock()


async def _handle_backoff(details: dict[str, Any]):
    limiter: AsyncLimiter = details["kwargs"]["_limiter"]
    logger.warning("HTTP error 429 received")
    current_time = datetime.now()
    decay_factor = 0.8
    global last_time_backoff, last_time_success
    async with lock:
        if (
            last_time_backoff is None
            or (current_time - last_time_backoff).total_seconds() >= 10
        ):
            logger.warning(
                f"Decreasing rate per second from {limiter._rate_per_sec} to {limiter._rate_per_sec * decay_factor}"
            )
            last_time_backoff = current_time
            limiter._rate_per_sec *= decay_factor


async def _handle_success(details: dict[str, Any]):
    limiter: AsyncLimiter = details["kwargs"]["_limiter"]
    growth_factor = 1.01
    current_time = datetime.now()
    original_rate = limiter.max_rate / limiter.time_period
    global last_time_success, last_time_backoff
    async with lock:
        if (
            (
                last_time_success is None
                or (current_time - last_time_success).total_seconds() >= 60
            )
            and (
                last_time_backoff is not None
                and (current_time - last_time_backoff).total_seconds() >= 60
            )
            and limiter._rate_per_sec < original_rate
        ):
            logger.debug(
                f"Increasing rate per second from {limiter._rate_per_sec} to {limiter._rate_per_sec * growth_factor}"
            )
            last_time_success = current_time
            new_rate = limiter._rate_per_sec * growth_factor
            limiter._rate_per_sec = min(new_rate, original_rate)


@on_exception(
    expo,
    ClientResponseError,
    max_tries=5,
    on_backoff=_handle_backoff,
    on_success=_handle_success,
    jitter=random_jitter,
    giveup=lambda e: e.status not in [429, 503],  # retry only on 429 and 503
    logger=logger,
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
    params: Union[dict[str, str], list[tuple[str, str]]],
    limiter: AsyncLimiter,
    index: int,
):
    """
    Runs a GET request against the provided URL's API with the provided params
    Args:
        url: target URL
        params: dictionary of HTTP parameters
        limiter: an aiolimiter AsyncLimiter
        index: index of the request

    Returns:
        a coroutine which provides the request\'s body in json format when awaited
    """
    try:
        response = None
        async with limiter:
            response = await _fetch(url=url, params=params, _limiter=limiter)
    except (CancelledError, ClientResponseError, JSONDecodeError) as e:
        raise e
    finally:
        return index, response
