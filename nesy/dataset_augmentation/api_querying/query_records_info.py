import heapq
from typing import Any, Union, Optional

from loguru import logger

from nesy.dataset_augmentation.api_querying.QueryResults import QueryResults
from .get_request_with_limiter import get_request_with_limiter
from .query_records_utils import extract_artist, extract_title, extract_year
from .score import compute_score_triple, push_to_heap
from .utils import (
    process_responses_with_joblib,
    get_async_limiter,
    process_http_requests,
    ErrorCode,
    encode_title,
)


def _extract_info(
    data: Union[
        tuple[tuple[str, list[str], Optional[str]], dict[str, Any], ErrorCode],
        tuple[tuple[str, list[str], Optional[str]], None, ErrorCode],
    ]
):
    query_tuple, response, err = data
    title_q, person_q, year_q = query_tuple
    # TODO check if more than one value from person_q is needed
    single_person = person_q[0] if len(person_q) > 0 else None
    person_r, year_r = None, None
    if err is not None:
        return title_q, person_q, year_q, err

    results_with_scores = []
    try:
        results = response["releases"]
    except KeyError as e:
        logger.error(f"Failed to retrieve '{title_q}' information due to error: {e}")
        return title_q, person_q, year_q, ErrorCode.JsonProcess
    if len(results) == 0:
        logger.debug(f"Title '{title_q}' had no matches")
        return title_q, person_q, year_q, ErrorCode.NotFound
    for result in results:
        title_r_i = extract_title(result)
        person_r_i, person_score = extract_artist(result, single_person)
        year_r_i = extract_year(result)
        score = compute_score_triple(
            (title_q, single_person, year_q), (title_r_i, person_r_i, year_r_i)
        )
        if score >= 0.4:
            push_to_heap(
                results_with_scores, (person_r_i, year_r_i), score + person_score
            )

    if len(results_with_scores) == 0:
        logger.debug(f"Title '{title_q}' had no matches")
        return title_q, person_q, year_q, ErrorCode.NotFound
    n_best = heapq.nlargest(1, results_with_scores)
    if len(n_best) > 0:
        best = n_best[0]
        person_r, year_r = best[-1]
    if person_q is None and person_r is None:
        logger.warning(f"Failed to retrieve artist for '{title_q}'")
    if year_q is None and year_r is None:
        logger.warning(f"Failed to retrieve year for '{title_q}'")

    if not isinstance(person_r, list):
        person_r = [person_r]

    return (
        title_q,
        person_r if person_q == [] else person_q,
        year_r if year_q is None else year_q,
        err,
    )


async def get_records_info(
    query_data: list[tuple[str, Union[str, None], Union[str, None]]]
) -> dict[str, QueryResults]:
    """
    Given a list of record titles, asynchronously queries the MusicBrainz v2 API
    Args:
        query_data: list of record titles to be searched

    Returns:
        a coroutine which provides all the responses' bodies\' in json format when awaited
    """
    limiter = get_async_limiter(how_many=len(query_data), max_rate=1, time_period=1)
    tasks = [
        (
            (title, person, year),
            get_request_with_limiter(
                url="https://musicbrainz.org/ws/2/release",
                params={"query": encode_title(title), "limit": 10, "fmt": "json"},
                limiter=limiter,
                index=i,
            ),
        )
        for i, (title, person, year) in enumerate(query_data)
    ]
    responses = await process_http_requests(
        tasks=tasks, tqdm_desc="Querying MusicBrainz..."
    )

    music_info = process_responses_with_joblib(responses=responses, fn=_extract_info)
    return {
        title: {
            "title": title,
            "person": person,
            "year": year,
            "err": err,
            "api_name": "MusicBrainz",
        }
        for title, person, year, err in music_info
    }
