from typing import Any

import regex as re


def _search_author_reg(desc: str):
    s = re.search(r"(?<=\bby\b\s)(\s?\b[A-Z][a-z]*\b)+", desc)
    if s:
        author = s.group(0)
        return author
    return None


def extract_book_author(item: dict[str, Any]):
    author = None
    if "description" in item:
        description = item["description"]
        author = _search_author_reg(description)
    if (
        author is None
        and "detailedDescription" in item
        and "articleBody" in item["detailedDescription"]
    ):
        description = item["detailedDescription"]["articleBody"]
        author = _search_author_reg(description)
    return author


def extract_book_year(item: dict[str, Any]):
    year = None
    try:
        detailed_description = item["detailedDescription"]["articleBody"]
        patterns = [
            re.compile(
                r"(?<=\b(released|published).*\b)(\b(19|20)\d{2}\b)", re.IGNORECASE
            ),
            re.compile(
                r"(?<=\b(is a).*\b)(\b(19|20)\d{2}\b)(?=.{0,15}book)", re.IGNORECASE
            ),
        ]
        for pattern in patterns:
            s = re.search(pattern, detailed_description)
            if s:
                year = s.group(0)
                break
    except KeyError:
        pass
    finally:
        return year
