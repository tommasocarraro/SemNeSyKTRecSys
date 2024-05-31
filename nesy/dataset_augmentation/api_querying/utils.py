import asyncio
import os
import signal
from asyncio import CancelledError
from enum import Enum
from json import JSONDecodeError
from typing import Sequence, Callable, Coroutine, Generator, Any

from aiohttp import ClientResponseError
from aiolimiter import AsyncLimiter
from joblib import delayed, Parallel
from loguru import logger
from tqdm.asyncio import tqdm

from nesy.dataset_augmentation import state


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
    elif eta < 1:
        eta *= 60
        unit = "second" if eta == 1 else "seconds"
    logger.info(
        f"Currently retrieving {how_many} items at a rate of {max_rate} per {time_period} "
        f"{'second' if time_period==1 else 'seconds'}. This will take approximately {eta:.2f} {unit}"
    )
    return AsyncLimiter(max_rate=max_rate, time_period=time_period)


def _add_signal_handlers():
    loop = asyncio.get_running_loop()

    def shutdown() -> None:
        logger.info("Interrupt received: cancelling all running async tasks")
        for task in asyncio.all_tasks(loop):
            if task is not asyncio.current_task(loop):
                task.cancel()
        state.GRACEFUL_EXIT = True

    for sig in [signal.SIGINT, signal.SIGTERM]:
        loop.add_signal_handler(sig, shutdown)


def _manually_abort(pbar: Generator[Any, Any, None]) -> None:
    pbar.close()
    os.kill(os.getpid(), signal.SIGINT)


async def process_http_requests(tasks: list[tuple[tuple, Coroutine]], tqdm_desc: str):
    _add_signal_handlers()

    aws = [task[1] for task in tasks]

    results = []
    for response in (
        pbar := tqdm.as_completed(aws, desc=tqdm_desc, dynamic_ncols=True)
    ):
        body, err = None, None
        try:
            req_index, body = await response
            query_tuple = tasks[req_index][0]
        except CancelledError:
            err = ErrorCode.Cancelled
            pbar.close()
        except ClientResponseError as e:
            if e.status == 404:
                logger.warning(f"Item not found: {e}")
                err = ErrorCode.NotFound
            elif e.status in [429, 503]:
                _manually_abort(pbar)
                logger.error(f"Too many requests to the server: {e}")
                err = ErrorCode.Throttled
            elif e.status >= 500:
                logger.warning(f"Server error: {e}")
                err = ErrorCode.RemoteServer
            elif e.status == 401:
                _manually_abort(pbar)
                logger.error(f"Unauthorized: {e}")
                err = ErrorCode.Unauthorized
            else:
                logger.error(f"Unknown error: {e}")
                err = ErrorCode.Unknown
        except JSONDecodeError as e:
            _manually_abort(pbar)
            logger.error(f"JSON decoding error: {e}")
            err = ErrorCode.JSONDecode
        finally:
            results.append((query_tuple, body, err))
    return results


class ErrorCode(Enum):
    NotFound = 1
    Cancelled = 2
    Throttled = 3
    Unauthorized = 4
    RemoteServer = 5
    JSONDecode = 6
    Unknown = 7
    JsonProcess = 8
