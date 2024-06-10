from typing import Any, Union, Optional
import heapq
from loguru import logger

from config import TMDB_API_KEY
from .get_request_with_limiter import get_request_with_limiter
from .utils import (
    process_responses_with_joblib,
    process_http_requests,
    get_async_limiter,
    ErrorCode,
)
from .query_movies_and_tv_utils import (
    extract_movie_year,
    extract_tv_year,
    extract_movie_director,
    extract_movie_id,
    extract_title,
)
from .score import compute_score_pair, push_to_heap
from ..query_apis import QueryResults


def _extract_search_info(
    data: Union[
        tuple[tuple[str, Optional[str], Optional[str]], dict[str, Any], ErrorCode],
        tuple[tuple[str, Optional[str], Optional[str]], None, ErrorCode],
    ]
):
    query_tuple, response, err = data
    title_q, person_q, year_q = query_tuple
    person_r, year_r, movie_id = None, None, None
    if err is not None:
        return title_q, person_q, year_q, None, err

    results_with_scores = []
    try:
        results = response["results"]
    except KeyError as e:
        logger.error(
            f"There was error while processing {title_q}'s search response: {e}"
        )
        return title_q, person_r, year_r, None, ErrorCode.JsonProcess
    if len(results) == 0:
        logger.debug(f"Title '{title_q}' had no matches")
        return title_q, person_q, year_q, None, ErrorCode.NotFound
    for result in results:
        try:
            media_type = result["media_type"]
            if media_type != "tv" and media_type != "movie":
                continue

            title_r_i = extract_title(result)
            if media_type == "tv":
                movie_id_i = None
                year_r_i = extract_tv_year(result)
            elif media_type == "movie":
                movie_id_i = extract_movie_id(result)
                year_r_i = extract_movie_year(result)
            else:
                raise ValueError("Unexpected media type")
            score = compute_score_pair((title_q, year_q), (title_r_i, year_r_i))
            if score >= 0.4:
                push_to_heap(results_with_scores, (year_r_i, movie_id_i), score)
        except KeyError:
            continue

    if len(results_with_scores) == 0:
        logger.debug(f"Title '{title_q}' had no matches")
        return title_q, person_q, year_q, ErrorCode.NotFound
    n_best = heapq.nlargest(1, results_with_scores)
    if len(n_best) > 0:
        best = n_best[0]
        year_r, movie_id = best[-1]
    if person_q is None and movie_id is None:
        logger.warning(f"Failed to retrieve director for '{title_q}'")
    if year_q is None and year_r is None:
        logger.warning(f"Failed to retrieve year for '{title_q}'")

    return (
        title_q,
        person_q,
        year_r if year_q is None else year_q,
        movie_id,
        err,
    )


def _extract_movie_info(
    data: Union[
        tuple[tuple[str, Optional[str], Optional[str]], dict[str, Any], ErrorCode],
        tuple[None, None],
    ]
):
    query, response, err = data
    person = None
    title = query[0]
    if err is not None:
        return title, person, err
    person = extract_movie_director(response)
    return title, person, err


async def get_movies_and_tv_info(
    query_data: list[tuple[str, Union[str, None], Union[str, None]]]
) -> dict[str, QueryResults]:
    """
    Given a list of book titles, asynchronously queries the TMDB API
    Args:
        query_data: list of movie and tv shows titles to be searched

    Returns:
        A coroutine which, if awaited, provides a dictionary containing the queried titles as keys and the
        retrieved information as values.
    """
    limiter = get_async_limiter(len(query_data), 35, time_period=1)

    # search TMDB
    search_tasks = [
        (
            (title, person, year),
            get_request_with_limiter(
                url="https://api.themoviedb.org/3/search/multi",
                params={
                    "query": title,
                    "api_key": TMDB_API_KEY,
                    "include_adult": "false",
                },
                limiter=limiter,
                index=i,
            ),
        )
        for i, (title, person, year) in enumerate(query_data)
    ]
    search_responses = await process_http_requests(
        search_tasks, "Searching for movies and TV series on TMDB..."
    )

    search_results = process_responses_with_joblib(
        responses=search_responses, fn=_extract_search_info
    )

    # extract all the results
    movies_results = [
        (title, movie_id)
        for title, person, _, movie_id, err in search_results
        if movie_id is not None and not err and person is None
    ]

    # query for all movies' directors
    movie_info_tasks = [
        (
            (title, movie_id),
            get_request_with_limiter(
                url=f"https://api.themoviedb.org/3/movie/{movie_id}/credits",
                params={"api_key": TMDB_API_KEY},
                limiter=limiter,
                index=i,
            ),
        )
        for i, (title, movie_id) in enumerate(movies_results)
    ]
    movies_info_responses = await process_http_requests(
        movie_info_tasks, "Retrieving movies directors from TMDB..."
    )
    movie_info = process_responses_with_joblib(
        responses=movies_info_responses, fn=_extract_movie_info
    )

    # assemble the output dictionary
    movies_and_tv_dict = {
        title: {
            "title": title,
            "person": person,
            "year": year,
            "err": err,
            "api_name": "The Movie Database",
        }
        for title, person, year, _, err in search_results
    }
    for title, person, err in movie_info:
        movies_and_tv_dict[title]["person"] = person
        if err is not None:
            movies_and_tv_dict["err"] = err
    return movies_and_tv_dict
