import heapq
import re
from typing import Any, Optional

import jaro
import regex
from loguru import logger

from config import GOOGLE_API_KEY
from .get_request import get_request
from .utils import (
    run_with_async_limiter,
    process_responses_with_joblib,
    get_async_limiter,
    tqdm_run_tasks_async,
)


def extract_info_music(res: Any):
    artist, year = None, None
    try:
        for item in res["itemListElement"]:
            base_body = item["result"]
            artist = base_body["description"].split("by")[1].strip()
            release_date = base_body["detailedDescription"]["articleBody"]
            pattern = r"\b(19|20)\d{2}\b"
            match = re.search(pattern, release_date)
            if match:
                year = match.group(0)
            break
    except IndexError as e:
        logger.error(f"Failed to retrieve the artist: {e}")
    except KeyError as e:
        logger.error(f"Failed to retrieve the data: {e}")
    except TypeError as e:
        logger.error(
            f"Type mismatch between the expected and actual json structure: {e}"
        )
    return artist, year


def extract_info_movie_tvseries(res: Any):
    director, year = None, None
    for item in res[1]["itemListElement"]:
        base_body = item["result"]
        detailed_description = base_body["detailedDescription"]["articleBody"]
        pattern = r"\b(19|20)\d{2}\b"
        match = re.search(pattern, detailed_description)
        if match:
            year = match.group(0)
        pattern = r"(?i)directed by ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
        match = re.search(pattern, detailed_description)
        if match:
            director = match.group(0).lstrip("directed by ")
        else:
            pattern = r"(?i)by ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
            match = re.search(pattern, detailed_description)
            if match:
                director = match.group(0).lstrip("by ")
        if director is not None and year is not None:
            return director, year
    return director, year


def extract_books_info(
    data: tuple[str, dict[str, Any]]
) -> tuple[str, Optional[str], Optional[str]]:
    author, year = None, None
    title, res = data
    results_with_scores = []
    counter = 0
    try:
        results = res["itemListElement"]
        for result in results:
            body = result["result"]
            score = 0
            try:
                description = body["description"]
                s = regex.search(r"\b(?<=by\s)(\w*\s\w*)\b", description)
                if s:
                    author = s.group(0)
            except KeyError:
                try:
                    detailed_description = body["detailedDescription"]["articleBody"]
                    s = regex.search(r"\b(?<=by\s)(\w*\s\w*)\b", detailed_description)
                    if s:
                        author = s.group(0)
                except KeyError:
                    logger.warning(f"Failed to retrieve {title}'s author")
            try:
                detailed_description = body["detailedDescription"]["articleBody"]
                s = regex.search(
                    r"(?<=([Rr]eleased|[Pp]ublished).*\b)(\d{4})", detailed_description
                )
                if s:
                    year = s.group(0)
            except KeyError:
                logger.warning(f"Failed to retrieve {title}'s year")
            try:
                name = body["name"]
                score = jaro.jaro_winkler_metric(title, name)
            except KeyError:
                logger.warning(
                    f"Failed to retrieve {title}'s name, using first result instead of computing the score"
                )
            try:
                # TODO dunno if this is actually useful
                types = body["@type"]
                for dtype in types:
                    if dtype not in ["Book", "Thing"]:
                        score = score - 0.5 if score >= 0.5 else 0
            except KeyError:
                logger.warning(f"Failed to retrieve {title}'s types")
            # required because if the first element of the tuple is equal to another, the heap will use the second
            # to compare and will throw an error if the types are not comparable
            counter += 1
            heapq.heappush(
                results_with_scores,
                (score, counter, (author, year)),
            )
    except KeyError as e:
        logger.error(f"Failed to retrieve {title}'s information due to error: {e}")

    best_score = heapq.nlargest(1, results_with_scores)
    if len(best_score) > 0:
        author, year = best_score[0][2]

    return title, author, year


async def google_kg_search(
    titles: list[str],
    query_types: list[str],
    language: str = "en",
):
    """
    Given a list of titles, asynchronously queries the Google Graph Search API.
    Args:
        titles: list of titles to be searched
        query_types: types of entities to be searched
        language: languages of the results, defaults to "en"

    Returns:
        a coroutine which provides all the responses\' bodies\' in json format when awaited
    """
    limiter = get_async_limiter(how_many=len(titles), max_rate=10, time_period=1)
    if "Book" in query_types:
        extract_info_cb = extract_books_info
    elif "CDs" in query_types:
        extract_info_cb = extract_info_music
    else:
        extract_info_cb = extract_info_movie_tvseries
    tasks = [
        run_with_async_limiter(
            limiter=limiter,
            fn=get_request,
            url="https://kgsearch.googleapis.com/v1/entities:search",
            title=title,
            params=[
                ("query", title),
                ("key", GOOGLE_API_KEY),
                ("limit", 10),
                ("indent", "True"),
                ("languages", language),
                *list(zip(["types" for _ in range(len(query_types))], query_types)),
            ],
        )
        for title in titles
    ]

    responses = await tqdm_run_tasks_async(tasks, desc="Querying Google KG...")

    info_list = process_responses_with_joblib(responses=responses, fn=extract_info_cb)
    return {
        title: {"title": title, "person": author, "year": year}
        for title, author, year in info_list
    }
