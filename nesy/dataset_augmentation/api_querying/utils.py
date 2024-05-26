import asyncio
import signal
from asyncio import CancelledError
from typing import Sequence, Coroutine, Callable

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


async def run_with_async_limiter(
    limiter: AsyncLimiter, fn: Callable, *args, **kwargs
) -> Coroutine:
    """
    Runs the *fn* asynchronous function inside the asynchronous context *limiter*
    Args:
        limiter: an AsyncLimiter
        fn: an asynchronous function

    Returns: a coroutine to be awaited
    """
    async with limiter:
        return await fn(*args, **kwargs)


def get_async_limiter(
    how_many: int, max_rate: float, time_period: float
) -> AsyncLimiter:
    eta = how_many * time_period / max_rate / 60
    unit = "minute(s)"
    if eta > 60:
        eta /= 60
        unit = "hour(s)"
    logger.info(
        f"Currently retrieving {how_many} items at a rate of {max_rate} per {time_period} second(s). This will "
        f"take approximately {eta:.2f} {unit}"
    )
    return AsyncLimiter(max_rate=max_rate, time_period=time_period)


async def tqdm_run_tasks_async(tasks: list, desc: str):
    responses = []
    loop = asyncio.get_event_loop()

    loop.add_signal_handler(
        signal.SIGINT, lambda: [task.cancel() for task in asyncio.all_tasks()]
    )
    loop.add_signal_handler(
        signal.SIGTERM, lambda: [task.cancel() for task in asyncio.all_tasks()]
    )
    for res in (pbar := tqdm.as_completed(tasks, desc=desc, dynamic_ncols=True)):
        try:
            responses.append(await res)
        except (ClientResponseError, CancelledError) as e:
            pbar.close()
            if isinstance(e, ClientResponseError):
                logger.error(f"Stopping early due to HTTP error: {e}")
            elif isinstance(e, CancelledError):
                logger.info(f"SIGINT detected. Quitting gracefully...")
            else:
                logger.error(f"An error occurred: {e}")
    return responses
