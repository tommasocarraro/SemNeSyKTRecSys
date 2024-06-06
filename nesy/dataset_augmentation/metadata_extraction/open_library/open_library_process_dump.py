import re
from typing import Callable, Any, Optional

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
    # check if title present and no more than 200 characters (Amazon title limit)
    if (
        "title" not in edition
        or edition["title"] is None
        or len(edition["title"]) > 200
    ):
        return None
    title = edition["title"]

    if "publish_date" not in edition:
        return None
    else:
        # extract year if available, discard if published after 2014
        publish_date = edition["publish_date"]
        s = re.search(r"((?:19|20)\d{2})", publish_date)
        if s:
            year = s.group(1)
            if int(year) > 2014:
                return None
        else:
            return None

    if (
        "authors" not in edition
        or len(edition["authors"]) == 0
        or len(edition["authors"][0]) > 30
    ):
        return None
    # extract list of authors if available, make sure the type is consistent
    authors = edition["authors"]
    if isinstance(authors[0], dict):
        authors = [auth_dict["key"].split("/")[2] for auth_dict in authors]
    else:
        authors = [auth.split("/")[2] for auth in authors]
    if "works" not in edition:
        return None
    works = edition["works"]
    works_keys = [work_dict["key"].split("/")[2] for work_dict in works]
    return {
        "key": edition["key"].split("/")[2],
        "title": title.strip().lower(),
        "year": year.strip(),
        "authors": authors,
        "works": works_keys,
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
    return {"key": author["key"].split("/")[2], "name": author["name"].strip().lower()}


def _process_work(work: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    Process a work object from the open library dump
    Args:
        work: work object

    Returns:
        Either the transformed object or None
    """
    title = work.get("title", "")
    if len(title) == 0 or len(title) > 200:
        return None
    # extract list of authors if available, make sure the type is consistent
    authors = work.get("authors", [])
    if 0 < len(authors) <= 30:
        if isinstance(authors[0], dict):
            authors = [
                auth_dict["author"]["key"].split("/")[2]
                for auth_dict in authors
                if "author" in auth_dict and "key" in auth_dict["author"]
            ]
        else:
            authors = [auth.split("/")[2] for auth in authors]
    else:
        return None
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
