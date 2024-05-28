from typing import Any, Union

from loguru import logger

from config import TMDB_API_KEY
from .get_request import get_request_with_limiter
from .utils import (
    process_responses_with_joblib,
    process_http_requests,
    get_async_limiter,
)


def _extract_search_info(data: Union[tuple[str, dict[str, Any]], tuple[str, None]]):
    item_id, year, err = None, None, False
    title, res = data
    err = False
    if res is None:
        err = True
    else:
        try:
            if len(res["results"]) == 0:
                logger.warning(f"No results for {title}")
                return title, (item_id, year)
            base_body = res["results"][0]
            media_type = base_body["media_type"]

            if media_type == "tv":
                try:
                    first_air_date = base_body["first_air_date"]
                    year = first_air_date.split("-")[0]
                except (KeyError, IndexError):
                    logger.warning(f"Failed to retrieve year for {title}")
                    year = None
            elif media_type == "movie":
                try:
                    item_id = base_body["id"]
                except KeyError:
                    logger.warning(f"Failed to retrieve item id for {title}")
                    item_id = None
                try:
                    release_date = base_body["release_date"]
                    year = release_date.split("-")[0]
                except (KeyError, IndexError):
                    logger.warning(f"Failed to retrieve year for {title}")
                    year = None
        except (TypeError, KeyError, IndexError) as e:
            logger.error(
                f"There was error while processing {title}'s search response: {e}"
            )
            err = True
    return title, (item_id, year), err


def _extract_movie_info(data: Union[tuple[str, dict[str, Any]], tuple[str, None]]):
    person = None
    title, res = data
    err = False
    if res is None:
        err = True
    else:
        try:
            crew = res["crew"]
            for crew_member in crew:
                if crew_member["job"] == "Director":
                    person = crew_member["name"]
                    break
        except TypeError as e:
            logger.error(
                f"There was error while processing {title}'s info response: {e}"
            )
            err = True
        except KeyError:
            logger.warning(f"Failed to retrieve {title}'s director")
    return title, person, err


async def get_movies_and_tv_info(titles: list[str]):
    """
    Given a list of book titles, asynchronously queries the TMDB API
    Args:
        titles: list of movie and tv shows titles to be searched

    Returns:
        A coroutine which, if awaited, provides a dictionary containing the queried titles as keys and the
        retrieved information as values.
    """
    limiter = get_async_limiter(len(titles), 35, time_period=1)

    # search TMDB
    search_tasks = [
        get_request_with_limiter(
            url="https://api.themoviedb.org/3/search/multi",
            params={
                "query": title,
                "api_key": TMDB_API_KEY,
                "include_adult": "false",
                "page": 1,
            },
            limiter=limiter,
            title=title,
        )
        for title in titles
    ]
    search_responses = await process_http_requests(
        search_tasks, "Searching for movies and TV series on TMDB..."
    )

    search_results = process_responses_with_joblib(
        responses=search_responses, fn=_extract_search_info
    )

    # extract all the results
    movies_results = [
        (title, movie)
        for title, movie, err in search_results
        if movie[0] is not None and not err
    ]

    # query for all movies' directors
    movie_info_tasks = [
        get_request_with_limiter(
            url=f"https://api.themoviedb.org/3/movie/{movie[0]}/credits",
            params={"api_key": TMDB_API_KEY},
            limiter=limiter,
            title=title,
        )
        for title, movie in movies_results
    ]
    movies_info_responses = await process_http_requests(
        movie_info_tasks, "Retrieving movies directors from TMDB..."
    )
    movie_info = process_responses_with_joblib(
        responses=movies_info_responses, fn=_extract_movie_info
    )

    # assemble the output dictionary
    movies_and_tv_dict = {
        title: {"title": title, "person": None, "year": data[1], "err": err}
        for title, data, err in search_results
    }
    for title, person, err in movie_info:
        movies_and_tv_dict[title]["person"] = person
        if err:
            movies_and_tv_dict["err"] = err
    return movies_and_tv_dict
