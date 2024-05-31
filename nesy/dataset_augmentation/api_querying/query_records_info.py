import heapq
from typing import Any, Union, Optional

from loguru import logger

from .get_request_with_limiter import get_request_with_limiter
from .utils import (
    process_responses_with_joblib,
    get_async_limiter,
    process_http_requests,
)
from .query_records_utils import extract_artist, extract_title, extract_year
from .score import compute_score_triple, push_to_heap


def _extract_info(
    data: Union[
        tuple[tuple[str, Optional[str], Optional[str]], dict[str, Any]],
        tuple[None, None],
    ]
):
    query, response = data
    if query is None:  # TODO fix up
        return None, None, None, None
    title_q, person_q, year_q = query
    person_r, year_r = None, None
    err = False
    if response is None:
        return title_q, person_q, year_q, True

    results_with_scores = []
    try:
        results = response["releases"]
    except KeyError:
        logger.info(f"No results found for {title_q}")
        return title_q, person_q, year_q, True
    for result in results:
        title_r_i = extract_title(result)
        person_r_i, person_score = extract_artist(result, person_q)
        year_r_i = extract_year(result)
        score = compute_score_triple(
            (title_q, person_q, year_q), (title_r_i, person_r_i, year_r_i)
        )
        push_to_heap(results_with_scores, (person_r_i, year_r_i), score + person_score)

    n_best = heapq.nlargest(1, results_with_scores)
    if len(n_best) > 0:
        best = n_best[0]
        person_r, year_r = best[-1]
    if person_q is None and person_r is None:
        logger.warning(f"Failed to retrieve {title_q}'s author")
    if year_q is None and year_r is None:
        logger.warning(f"Failed to retrieve {title_q}'s year")

    return (
        title_q,
        person_r if person_q is None else person_q,
        year_r if year_q is None else year_q,
        err,
    )


async def get_records_info(
    query_data: list[tuple[str, Union[str, None], Union[str, None]]]
):
    """
    Given a list of record titles, asynchronously queries the MusicBrainz v2 API
    Args:
        query_data: list of record titles to be searched

    Returns:
        a coroutine which provides all the responses' bodies\' in json format when awaited
    """
    limiter = get_async_limiter(how_many=len(query_data), max_rate=1, time_period=1)
    tasks = [
        get_request_with_limiter(
            url="https://musicbrainz.org/ws/2/release",
            title=title,
            params={"query": title, "limit": 10, "fmt": "json"},
            limiter=limiter,
            person=person,
            year=year,
        )
        for title, person, year in query_data
    ]
    responses = await process_http_requests(
        tasks=tasks, tqdm_desc="Querying MusicBrainz..."
    )

    music_info = process_responses_with_joblib(responses=responses, fn=_extract_info)
    return {
        title: {"title": title, "person": person, "year": year, "err": err}
        for title, person, year, err in music_info
    }
