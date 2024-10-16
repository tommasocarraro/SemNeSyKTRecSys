import heapq
from typing import Any, Union, Optional

from loguru import logger

from config import GOOGLE_API_KEY
from .get_request_with_limiter import get_request_with_limiter
from .query_google_kg_search_utils import extract_book_author, extract_book_year
from .score import compute_score_triple, push_to_heap
from .utils import (
    process_responses_with_joblib,
    get_async_limiter,
    process_http_requests,
    ErrorCode,
)


def _extract_books_info(
    data: Union[
        tuple[tuple[str, Optional[str], Optional[str]], dict[str, Any], ErrorCode],
        tuple[tuple[str, Optional[str], Optional[str]], None, ErrorCode],
    ]
):
    query_tuple, response, err = data
    title_q, person_q, year_q = query_tuple
    person_r, year_r = None, None
    if err is not None:
        return title_q, person_q, year_q, err

    results_with_scores = []

    try:
        results = response["itemListElement"]
    except KeyError as e:
        logger.error(f"Failed to retrieve '{title_q}' information due to error: {e}")
        return title_q, person_q, year_q, ErrorCode.JsonProcess
    if len(results) == 0:
        logger.debug(f"Title '{title_q}' had no matches")
        return title_q, person_q, year_q, ErrorCode.NotFound
    for result in results:
        try:
            body = result["result"]
            title_r_i = body["name"]
            person_r_i = extract_book_author(body)
            year_r_i = extract_book_year(body)
            score = compute_score_triple(
                (title_q, person_q, year_q), (title_r_i, person_r_i, year_r_i)
            )
            if score >= 0.4:
                push_to_heap(results_with_scores, (person_r_i, year_r_i), score)
        except KeyError:
            continue

    if len(results_with_scores) == 0:
        logger.debug(f"Title '{title_q}' had no matches")
        return title_q, person_q, year_q, ErrorCode.NotFound
    n_best = heapq.nlargest(1, results_with_scores)
    if len(n_best) > 0:
        best = n_best[0]
        person_r, year_r = best[-1]
    if person_q is None and person_r is None:
        logger.warning(f"Failed to retrieve author for '{title_q}'")
    if year_q is None and year_r is None:
        logger.warning(f"Failed to retrieve year for '{title_q}'")

    return (
        title_q,
        person_r if person_q is None else person_q,
        year_r if year_q is None else year_q,
        err,
    )


async def google_kg_search(
    query_data: list[tuple[str, Union[str, None], Union[str, None]]],
    query_types: list[str],
    language: str = "en",
):
    """
    Given a list of titles, asynchronously queries the Google Graph Search API.
    Args:
        query_data: list of titles to be searched
        query_types: types of entities to be searched
        language: languages of the results, defaults to "en"

    Returns:
        a coroutine which provides all the responses\' bodies\' in json format when awaited
    """
    limiter = get_async_limiter(how_many=len(query_data), max_rate=9, time_period=1)

    tasks = [
        (
            (title, person, year),
            get_request_with_limiter(
                limiter=limiter,
                url="https://kgsearch.googleapis.com/v1/entities:search",
                params=[
                    ("query", title),
                    ("key", GOOGLE_API_KEY),
                    ("limit", 10),
                    ("indent", "True"),
                    ("languages", language),
                    *list(zip(["types" for _ in range(len(query_types))], query_types)),
                ],
                index=i,
            ),
        )
        for i, (title, person, year) in enumerate(query_data)
    ]
    responses = await process_http_requests(tasks, tqdm_desc="Querying Google KG...")

    info_list = process_responses_with_joblib(
        responses=responses, fn=_extract_books_info
    )
    return {
        title: {"title": title, "person": author, "year": year, "err": err}
        for title, author, year, err in info_list
        if title is not None
    }
