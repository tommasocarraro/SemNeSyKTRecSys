from typing import Any, Union

import regex as re


def get_nested_value(dictionary, keys):
    value = dictionary
    for key in keys:
        value = value[key]
    return value


def extract_book_author(item: dict[str, Any], desc_field: Union[str, list[str]]):
    author = None
    try:
        if isinstance(desc_field, str):
            description = item[desc_field]
        else:
            description = get_nested_value(item, desc_field)
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
