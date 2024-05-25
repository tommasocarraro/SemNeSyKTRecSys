from typing import Any

from aiolimiter import AsyncLimiter
from loguru import logger

from config import TMDB_API_KEY
from .get_request import get_request
from .utils import (
    run_with_async_limiter,
    process_responses_with_joblib,
    tqdm_run_tasks_async,
)


async def get_movies_and_tv_info(titles: list[str]):
    """
    Given a list of book titles, asynchronously queries the TMDB API
    Args:
        titles: list of movie and tv shows titles to be searched

    Returns:
        A coroutine which, if awaited, provides a dictionary containing the queried titles as keys and the
        retrieved information as values.
    """
    limiter = AsyncLimiter(35, time_period=1)

    search_tasks = [
        run_with_async_limiter(
            limiter=limiter,
            fn=get_request,
            title=title,
            url="https://api.themoviedb.org/3/search/multi",
            params={
                "query": title,
                "api_key": TMDB_API_KEY,
                "include_adult": "false",
                "page": 1,
            },
        )
        for title in titles
    ]

    search_responses: list[tuple[str, dict[str, Any]]] = await tqdm_run_tasks_async(
        search_tasks, "Searching for movies and TV series on TMDB..."
    )

    def extract_search_info(data: tuple[str, dict[str, Any]]):
        item_id, year = None, None
        title, res = data
        try:
            if len(res["results"]) == 0:
                logger.error(f"No results for {title}")
                return title, (item_id, year)
            base_body = res["results"][0]
            media_type = base_body["media_type"]

            if media_type == "tv":
                try:
                    first_air_date = base_body["first_air_date"]
                    year = first_air_date.split("-")[0]
                except (KeyError, IndexError):
                    logger.error(f"Failed to retrieve year for {title}")
                    year = None
            elif media_type == "movie":
                try:
                    item_id = base_body["id"]
                except KeyError:
                    logger.error(f"Failed to retrieve item id for {title}")
                    item_id = None
                try:
                    release_date = base_body["release_date"]
                    year = release_date.split("-")[0]
                except (KeyError, IndexError):
                    logger.error(f"Failed to retrieve year for {title}")
                    year = None
        except (TypeError, KeyError, IndexError) as e:
            logger.error(
                f"There was error while processing {title}'s search response: {e}"
            )
        return title, (item_id, year)

    search_results = process_responses_with_joblib(
        responses=search_responses, fn=extract_search_info
    )

    movies_and_tv_dict = {
        title: {"title": title, "person": None, "year": data[1]}
        for title, data in search_results
    }

    movies_results = [
        (title, movie) for title, movie in search_results if movie[0] is not None
    ]

    movie_info_tasks = [
        run_with_async_limiter(
            limiter=limiter,
            fn=get_request,
            title=title,
            url=f"https://api.themoviedb.org/3/movie/{movie[0]}/credits",
            params={"api_key": TMDB_API_KEY},
        )
        for title, movie in movies_results
    ]

    movies_info_responses = await tqdm_run_tasks_async(
        movie_info_tasks, "Retrieving movies directors from TMDB..."
    )

    def extract_movie_info(data: tuple[str, dict[str, Any]]):
        person = None
        title, res = data
        try:
            crew = res["crew"]
            for crew_member in crew:
                if crew_member["job"] == "Director":
                    person = crew_member["name"]
                    break
        except (TypeError, KeyError, IndexError) as e:
            logger.error(
                f"There was error while processing {title}'s info response: {e}"
            )
        return title, person

    movie_info = process_responses_with_joblib(
        responses=movies_info_responses, fn=extract_movie_info
    )

    for title, person in movie_info:
        movies_and_tv_dict[title]["person"] = person
    return movies_and_tv_dict
