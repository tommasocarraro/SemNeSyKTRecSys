import asyncio
import signal
from json import JSONDecodeError
from typing import Sequence, Callable

from aiohttp import ClientResponseError
from aiolimiter import AsyncLimiter
from joblib import delayed, Parallel
from loguru import logger
from tqdm.asyncio import tqdm


def process_responses_with_joblib(
    fn: Callable, responses: Sequence, n_jobs: int = -1
) -> list:
    pool = Parallel(n_jobs=n_jobs, backend="loky")
    return pool(delayed(fn)(res) for res in responses)


def get_async_limiter(
    how_many: int, max_rate: float, time_period: float
) -> AsyncLimiter:
    """
    Logs ETA with the given parameters and returns the asynchronous limiter
    Args:
        how_many: how many total calls
        max_rate: how many calls per *time_period*
        time_period: time period, measured in seconds

    Returns:
        An asynchronous rate limiter
    """
    eta = how_many * time_period / max_rate / 60
    unit = "minute" if eta == 1 else "minutes"
    if eta > 60:
        eta /= 60
        unit = "hour" if eta == 1 else "hours"
    logger.info(
        f"Currently retrieving {how_many} items at a rate of {max_rate} per {time_period} {'second' if time_period==1 else 'seconds'}. This will "
        f"take approximately {eta:.2f} {unit}"
    )
    return AsyncLimiter(max_rate=max_rate, time_period=time_period)


def _stop_querying():
    for task in asyncio.all_tasks():
        task.cancel()


async def process_http_requests(tasks: list, tqdm_desc: str):
    loop = asyncio.get_event_loop()

    for sig in [signal.SIGINT, signal.SIGTERM]:
        loop.add_signal_handler(sig, _stop_querying)

    results = []
    for res in (pbar := tqdm.as_completed(tasks, desc=tqdm_desc, dynamic_ncols=True)):
        try:
            result = await res
            results.append(result)
        except (ClientResponseError, JSONDecodeError) as e:
            if isinstance(e, ClientResponseError):
                if e.status == 404:
                    logger.warning(f"Item not found: {e}")
                elif e.status in [429, 503]:
                    logger.error(f"Too many requests to the server: {e}")
                    _stop_querying()
                    pbar.close()
                elif e.status >= 500:
                    logger.warning(f"Server error: {e}")
                elif e.status == 401:
                    logger.error(f"Unauthorized: {e}")
                    _stop_querying()
                    pbar.close()
                else:
                    logger.error(f"Unknown error: {e}")
            elif isinstance(e, JSONDecodeError):
                logger.error(f"Failed to parse response: {e}")
    return results
