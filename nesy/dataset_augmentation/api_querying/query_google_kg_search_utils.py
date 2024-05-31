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
                r"(?<=(\bis\b\s\ba\b)\s)(\b(19|20)\d{2}\b)(?=.{0,25}(?:book|collection|novel))",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?:\bwritten\b\s\bby\b\s(?:\b\w+\b\s){0,20}\bin\b\s)(\b(19|20)\d{2}\b)",
                re.IGNORECASE,
            ),
            re.compile(r"(\b(19|20)\d{2}\b)"),  # fallback, probably inaccurate
        ]
        for i, pattern in enumerate(patterns):
            s = re.search(pattern, detailed_description)
            if (
                s
                and (i == len(patterns) - 1 and len(s.groups()) == 1)
                or i != len(patterns) - 1
            ):
                year = s.group(0)
                break
    except KeyError:
        pass
    finally:
        return year
