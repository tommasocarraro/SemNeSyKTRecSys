from typing import Any
import regex as re


def extract_book_author(item: dict[str, Any], desc_field: str):
    author = None
    try:
        description = item[desc_field]
        s = re.search(r"\b(?<=by\s)(\w*\s\w*)\b", description)
        if s:
            author = s.group(0)
    except KeyError:
        pass
    finally:
        return author


def extract_book_year(item: dict[str, Any]):
    year = None
    try:
        detailed_description = item["detailedDescription"]
        s = re.search(
            r"(?<=([Rr]eleased|[Pp]ublished).*\b)(\d{4})", detailed_description
        )
        if s:
            year = s.group(0)
    except KeyError:
        pass
    finally:
        return year
