import heapq
from typing import Any, Union

import jaro
from loguru import logger

from config import GOOGLE_API_KEY
from .get_request_with_limiter import get_request_with_limiter
from .query_google_kg_search_utils import extract_book_author, extract_book_year
from .utils import (
    process_responses_with_joblib,
    get_async_limiter,
    process_http_requests,
)


def _extract_books_info(data: Union[tuple[str, dict[str, Any]], tuple[str, None]]):
    author, year = None, None
    title, res = data
    err = False
    if res is None:
        err = True
    else:
        results_with_scores = []
        counter = 0
        try:
            results = res["itemListElement"]
            for result in results:
                body = result["result"]
                author = extract_book_author(body, "description")
                if author is None:
                    author = extract_book_author(
                        body, ["detailedDescription", "articleBody"]
                    )
                year = extract_book_year(body)
                try:
                    name = body["name"]
                    score = jaro.jaro_winkler_metric(title, name)
                except KeyError:
                    continue
                try:
                    # TODO dunno if this is actually useful
                    types = body["@type"]
                    for dtype in types:
                        if dtype not in ["Book", "Thing"]:
                            score = max(score - 0.1, 0)
                except KeyError:
                    pass
                # counter is required because if the first element of the tuple is equal to another, the heap will
                # use the second to compare and will throw an error if the types are not comparable
                counter += 1
                heapq.heappush(results_with_scores, (score, counter, (author, year)))
        except KeyError as e:
            logger.error(f"Failed to retrieve {title}'s information due to error: {e}")
            err = True

        n_best = heapq.nlargest(1, results_with_scores)
        if len(n_best) > 0:
            best = n_best[0]
            author, year = best[-1]
        if author is None:
            logger.warning(f"Failed to retrieve {title}'s author")
        if year is None:
            logger.warning(f"Failed to retrieve {title}'s year")

    return title, author, year, err


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
    tasks = [
        get_request_with_limiter(
            limiter=limiter,
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
    responses = await process_http_requests(tasks, tqdm_desc="Querying Google KG...")

    info_list = process_responses_with_joblib(
        responses=responses, fn=_extract_books_info
    )
    return {
        title: {"title": title, "person": author, "year": year, "err": err}
        for title, author, year, err in info_list
    }
