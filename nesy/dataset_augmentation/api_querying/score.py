import heapq
from typing import Union, Optional

from jaro import jaro_winkler_metric


def _is_year_match(year_q: Optional[str], year_r: Optional[str]) -> bool:
    return year_q is not None and year_r is not None and year_q == year_r


def _is_person_match(person_q: Optional[str], person_r: Optional[str]) -> bool:
    return (
        person_q is not None
        and person_r is not None
        and jaro_winkler_metric(person_q, person_r) >= 0.8
    )


def compute_score_pair(
    query: tuple[str, Union[str, None]], response: tuple[str, Union[str, None]]
):
    title_q, year_q = query
    title_r, year_r = response
    if not _is_year_match(year_q, year_r):
        return 0
    return jaro_winkler_metric(title_q, title_r)


def compute_score_triple(
    query: tuple[str, Union[str, None], Union[str, None]],
    response: tuple[str, Union[str, None], Union[str, None]],
) -> float:
    title_q, person_q, year_q = query
    title_r, person_r, year_r = response

    if not _is_person_match(person_q, person_r) or not _is_year_match(year_q, year_r):
        return 0

    return jaro_winkler_metric(title_q, title_r)


counter = 0


def push_to_heap(
    heap: list,
    item: tuple,
    score: float,
) -> None:
    global counter
    # counter is required because if the first element of the tuple is equal to another, the heap will
    # use the second to compare and will throw an error if the types are not comparable
    counter += 1
    heapq.heappush(heap, (score, counter, item))
