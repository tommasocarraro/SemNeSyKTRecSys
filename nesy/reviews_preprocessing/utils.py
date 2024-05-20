from typing import Any, Sequence, Coroutine, Callable

from aiolimiter import AsyncLimiter
from joblib import delayed, Parallel


def process_responses_with_joblib(
    fn: Callable, responses: Sequence, n_jobs: int = -1
) -> list:
    pool = Parallel(n_jobs=n_jobs, backend="loky")
    return pool(delayed(fn)(res) for res in responses)


async def run_with_async_limiter(
    limiter: AsyncLimiter, fn: Callable, *args, **kwargs
) -> Coroutine:
    """
    Runs the *fn* function inside the asynchronous context *limiter*
    Args:
        limiter: an AsyncLimiter
        fn: an asynchronous function

    Returns: a coroutine to be awaited
    """
    async with limiter:
        return await fn(*args, **kwargs)
