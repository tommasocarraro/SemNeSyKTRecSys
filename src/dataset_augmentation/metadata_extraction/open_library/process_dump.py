import re
from typing import Any, Callable, Optional

import orjson as json
from tqdm.auto import tqdm


def _process_open_library_file(
    in_file_path: str,
    out_file_path: str,
    process_obj_fn: Callable[[dict[str, Any]], Optional[dict[str, Any]]],
    tqdm_desc: str,
) -> None:
    """
    Process an open library dump file by extracting the JSON objects in the txt lines
    and transforming them according to function **process_obj**
    Args:
        in_file_path: open library dump file path
        out_file_path: where to save the output JSONL file
        process_obj_fn: object transformation function
        tqdm_desc: tqdm progress bar description

    Returns:
        None
    """
    with open(in_file_path, "r", encoding="utf-8") as in_file:
        with open(out_file_path, "wb") as out_file:
            for line in tqdm(in_file, desc=tqdm_desc, dynamic_ncols=True):
                obj_type, key, _, _creation_date, string_object = line.split("\t")
                obj = json.loads(string_object)
                processed_obj = process_obj_fn(obj)
                if processed_obj is not None:
                    out_file.write(json.dumps(processed_obj) + b"\n")


def _process_edition(edition: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    Process an edition object from the open library dump
    Args:
        edition: edition object

    Returns:
        Either the transformed object or None
    """
    if (
        "title" not in edition
        or edition["title"] is None
        or len(edition["title"]) > 200
    ):
        return None
    title = edition["title"]

    year = None
    if "publish_date" in edition:
        publish_date = edition["publish_date"]
        s = re.search(r"((?:19|20)\d{2})", publish_date)
        if s:
            year = s.group(1)

    authors = edition["authors"] if "authors" in edition else None
    if authors is not None and len(authors) > 0:
        if isinstance(authors[0], dict):
            authors = [auth_dict["key"].split("/")[2] for auth_dict in authors]
        else:
            authors = [auth.split("/")[2] for auth in authors]

    works = edition["works"] if "works" in edition else None
    if works is not None:
        works_keys = [work_dict["key"].split("/")[2] for work_dict in works]
    else:
        works_keys = None

    isbns = edition["isbn_10"] if "isbn_10" in edition else None
    return {
        "key": edition["key"].split("/")[2],
        "title": title,
        "title_query": title.lower().strip(),
        "year": year,
        "authors": authors,
        "works": works_keys,
        "isbns": isbns,
    }


def _process_author(author: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    Process an author object from the open library dump
    Args:
        author: author object

    Returns:
        Either the transformed object or None
    """
    # if author's name is unknown, skip it
    if "name" not in author:
        return None

    return {
        "key": author["key"].split("/")[2],
        "name": author["name"],
        "name_query": author["name"].lower().strip(),
    }


def _process_work(work: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    Process a work object from the open library dump
    Args:
        work: work object

    Returns:
        Either the transformed object or None
    """
    title = work["title"] if "title" in work and len(work["title"]) < 200 else None

    authors = work["authors"] if "authors" in work else None
    if authors is not None:
        if isinstance(authors[0], dict):
            authors = [
                auth_dict["author"]["key"].split("/")[2]
                for auth_dict in authors
                if "author" in auth_dict and "key" in auth_dict["author"]
            ]
        else:
            authors = [auth.split("/")[2] for auth in authors]

    key = work["key"].split("/")[2]
    return {"key": key, "title": title, "authors": authors}


def process_dump(
    editions_in_file_path: str,
    editions_out_file_path: str,
    authors_in_file_path: str,
    authors_out_file_path: str,
    works_in_file_path: str,
    works_out_file_path: str,
) -> None:
    """
    Process the open library editions and authors dump files
    Args:
        editions_in_file_path: open library editions dump file path
        editions_out_file_path: open library editions output JSONL file path
        authors_in_file_path: open library authors dump file path
        authors_out_file_path: open library authors output JSONL file path
        works_in_file_path: open library works dump file path
        works_out_file_path: open library works output JSONL file path

    Returns:
        None
    """
    _process_open_library_file(
        editions_in_file_path,
        editions_out_file_path,
        _process_edition,
        "Processing editions dump...",
    )

    _process_open_library_file(
        authors_in_file_path,
        authors_out_file_path,
        _process_author,
        "Processing authors dump...",
    )

    _process_open_library_file(
        works_in_file_path,
        works_out_file_path,
        _process_work,
        "Processing works dump...",
    )
