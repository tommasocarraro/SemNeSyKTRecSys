from typing import Coroutine

import aiohttp
import backoff
import requests


@backoff.on_exception(
    backoff.expo, requests.exceptions.RequestException, max_time=120, max_tries=10
)
async def get_request(base_url: str, params: dict[str, str]) -> Coroutine:
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
