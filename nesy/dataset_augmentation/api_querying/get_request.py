from typing import Coroutine

import aiohttp
import requests
from backoff import on_exception, expo, full_jitter


@on_exception(
    expo,
    requests.exceptions.RequestException,
    max_time=120,
    max_tries=10,
    jitter=full_jitter,
)
async def _get_request(base_url: str, params: dict[str, str]) -> Coroutine:
    """
    Performs an asynchronous HTTP GET request
    Args:
        base_url: API endpoint to contact
        params: URL parameters to attach to the base URL

    Returns: a coroutine which provides the request's json output when awaited
    """
    headers = {
        "accept": "application/json",
        "auth-agent": "SemNeSyKTRecSys-Python",
        "Accept-Charset": "utf-8",
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(
            base_url, params=params, headers=headers, raise_for_status=True
        ) as response:
            return await response.json()


async def get_request(url: str, title: str, params: dict[str, str]):
    """
    Runs a GET request against the provided URL's API with the provided params
    Args:
        url: target URL
        title: title to be searched, only used for debugging purposes
        params: dictionary of HTTP parameters

    Returns: a coroutine which provides the request's body in json format when awaited
    """
    try:
        return title, await _get_request(url, params)
    except requests.RequestException as e:
        print(f"There was an error while retrieving item {title}: {e}")
