import sys
from typing import Sequence, Coroutine, Callable

from aiohttp import ClientResponseError
from aiolimiter import AsyncLimiter
from joblib import delayed, Parallel
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
    print(
        f"Currently retrieving {how_many} items at a rate of {max_rate} per {time_period} second(s). This will "
        f"take approximately {eta:.2f} minutes"
    )
    return AsyncLimiter(max_rate=max_rate, time_period=time_period)


async def tqdm_process_responses(tasks: list):
    responses = []
    error = None
    for res in tqdm.as_completed(tasks, desc="Querying the API...", dynamic_ncols=True):
        try:
            responses.append(await res)
        except (ClientResponseError, KeyboardInterrupt) as e:
            error = e
            break
    # printing the error outside the loop to avoid messing up tqdm's progress bar
    if isinstance(error, ClientResponseError):
        print(f"Stopping early due to HTTP error: {error}", file=sys.stderr)
    else:
        print(f"Keyboard interrupt detected. Quitting gracefully...", file=sys.stderr)
    return responses
