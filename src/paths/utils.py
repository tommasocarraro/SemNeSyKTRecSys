import subprocess
from typing import Any, Literal

import pandas as pd
from loguru import logger


def get_rating_stats(rating_file: str, entity: Literal["item", "user"]) -> dict:
    """
    This function takes as input a rating file and counts the number of ratings for each user or item.
    The stats are returned in a dict.

    :param rating_file: path to rating file
    :param entity: whether to count the number of ratings of users or items
    :return: dictionary reporting the stats
    """
    if entity == "item":
        df_col = "itemId"
    elif entity == "user":
        df_col = "userId"
    else:
        raise ValueError("Entity must be 'item' or 'user'")
    df = pd.read_csv(rating_file)
    ratings_count = df.groupby(df_col).size().to_dict()
    # sort dictionary by the rating count
    return dict(sorted(ratings_count.items(), key=lambda entity_: entity_[1]))


def get_cold_start_items(stats: dict, threshold: int) -> list:
    """
    This function takes statistics about ratings in the dataset and returns the list of items (or users)
    that have less than or equal threshold ratings.

    :param stats: dictionary containing stats about ratings
    :param threshold: the threshold to select items (or users)
    :return: list of items (or users) with less than or equal threshold ratings
    """
    return [id_ for id_, count in stats.items() if count <= threshold]


def get_popular_items(stats: dict, threshold: int) -> list:
    """
    This function takes statistics about ratings in the dataset and returns the list of items (or users)
    that have more than or equal threshold ratings.

    :param stats: dictionary containing stats about ratings
    :param threshold: the threshold to select items (or users)
    :return: list of items (or users) with more than or equal threshold ratings
    """
    return [id_ for id_, count in stats.items() if count >= threshold]


def refine_cold_start_items(
    cold_start_list: list, target_mapping: dict[str, Any]
) -> list:
    """
    This function takes a cold-start list of items and refines it by removing items that have not been matched with
    Wikidata.

    :param cold_start_list: list of cold-start items
    :param target_mapping: mapping containing the matches in the target domain
    :return: refined cold-start list
    """
    return [
        target_mapping[id_]["wiki_id"]
        for id_ in cold_start_list
        if isinstance(target_mapping[id_], dict)
    ]


def refine_popular_items(popular_list: list, source_mapping: dict[str, Any]) -> list:
    """
    This function takes a cold-start list of items and refines it by removing items that have not been matched with
    Wikidata.

    :param popular_list: list of popular items
    :param source_mapping: mapping containing the matches in the source domain
    :return: refined popular list
    """
    return [
        source_mapping[id_]["wiki_id"]
        for id_ in popular_list
        if isinstance(source_mapping[id_], dict)
    ]


def run_shell_command(args: list[str], use_sudo=False) -> None:
    """
    Runs a unix shell command, possibly with admin privileges.

    :param args: shell command to run, split into a list of strings
    :param use_sudo: whether to run the command as sudo
    """
    if use_sudo:
        args.insert(0, "sudo")
    with subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ) as process:
        try:
            # Iterate over stdout line by line
            for line in process.stdout:
                print(line, end="")  # Print each line as it appears in real-time
            # Wait for the process to complete and capture the return code
            process.wait()

            # If an error occurs (non-zero exit), print stderr
            if process.returncode != 0:
                logger.error(process.stderr.read())

        except Exception as e:
            logger.error(e)
            exit(1)
