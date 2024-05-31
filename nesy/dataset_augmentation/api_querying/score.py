import heapq
from typing import Union

from jaro import jaro_winkler_metric


def compute_score(
    query: tuple[str, Union[str, None], Union[str, None]],
    response: tuple[str, Union[str, None], Union[str, None]],
) -> float:
    title_q, person_q, year_q = query
    title_r, person_r, year_r = response

    author_no_match = (
        person_q is not None
        and person_r is not None
        and jaro_winkler_metric(person_q, person_r) < 0.8
    )
    year_no_match = year_q is not None and year_r is not None and year_q != year_r
    if author_no_match or year_no_match:
        return 0

    return jaro_winkler_metric(title_q, title_r)


counter = 0


def push_to_heap(
    heap: list[tuple[float, int, tuple[Union[str, None], Union[str, None]]]],
    item: tuple[Union[str, None], Union[str, None]],
    score: float,
) -> None:
    global counter
    # counter is required because if the first element of the tuple is equal to another, the heap will
    # use the second to compare and will throw an error if the types are not comparable
    counter += 1
    heapq.heappush(heap, (score, counter, item))
