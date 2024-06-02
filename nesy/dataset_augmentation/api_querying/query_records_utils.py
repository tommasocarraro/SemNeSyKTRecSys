from typing import Any, Optional
import heapq
from jaro import jaro_winkler_metric


def extract_artist(item: dict[str, Any], person_q: str) -> tuple[Optional[str], float]:
    artists = item["artist-credit"]
    names_with_scores = []
    if person_q is None:
        person_r_i = artists[0]["name"]
        return person_r_i, 0
    else:
        for artist in artists:
            person_r_i = artist["name"]
            score = jaro_winkler_metric(person_q, person_r_i)
            names_with_scores.append((person_r_i, score))
    n_best = heapq.nlargest(1, names_with_scores)
    if len(n_best) > 0:
        person_r_i, score = n_best[0]
        return person_r_i, score
    return None, 0


def extract_title(item: dict[str, Any]) -> str:
    return item["title"]


def extract_year(item: dict[str, Any]) -> Optional[str]:
    year = None
    if "date" in item:
        release_date = item["date"]
        year = release_date.split("-")[0]
    elif "release-events" in item:
        release_events = item["release-events"]
        year = min(
            [release_event["date"].split("-")[0] for release_event in release_events]
        )
    return year
