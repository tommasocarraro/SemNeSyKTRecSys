import json
import subprocess
from typing import Any, Literal, Optional

import pandas as pd
import tqdm
from joblib import Parallel
from loguru import logger


def get_rating_stats(rating_file: str, entity: Literal["item", "user"], implicit=True) -> dict:
    """
    This function takes as input a rating file and counts the number of ratings for each user or item.
    The stats are returned in a dict.

    :param rating_file: path to rating file
    :param entity: whether to count the number of ratings of users or items
    :param implicit: whether the dataset contains implicit feedback
    :return: dictionary reporting the stats
    """
    if entity == "item":
        df_col = "itemId"
    elif entity == "user":
        df_col = "userId"
    else:
        raise ValueError("Entity must be 'item' or 'user'")
    df = pd.read_csv(rating_file)
    if implicit:
        df["rating"] = (df["rating"] >= 4).astype(int)
        df = df[df['rating'] == 1]
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


class ParallelTqdm(Parallel):
    """joblib.Parallel, but with a tqdm progressbar
    Additional parameters:
    ----------------------
    total_tasks: int, default: None
        the number of expected jobs. Used in the tqdm progressbar.
        If None, try to infer from the length of the called iterator, and
        fallback to use the number of remaining items as soon as we finish
        dispatching.
        Note: use a list instead of an iterator if you want the total_tasks
        to be inferred from its length.
    desc: str, default: None
        the description used in the tqdm progressbar.
    disable_progressbar: bool, default: False
        If True, a tqdm progressbar is not used.
    show_joblib_header: bool, default: False
        If True, show joblib header before the progressbar.
    """

    def __init__(
        self,
        *,
        total_tasks: Optional[int] = None,
        desc: Optional[str] = None,
        disable_progressbar=False,
        show_joblib_header=False,
        **kwargs
    ):
        if "verbose" in kwargs:
            raise ValueError(
                "verbose is not supported. "
                "Use show_progressbar and show_joblib_header instead."
            )
        super().__init__(verbose=(1 if show_joblib_header else 0), **kwargs)
        self.total_tasks = total_tasks
        self.desc = desc
        self.disable_progressbar = disable_progressbar
        self.progress_bar: Optional[tqdm.tqdm] = None

    def __call__(self, iterable):
        try:
            if self.total_tasks is None:
                # try to infer total_tasks from the length of the called iterator
                try:
                    self.total_tasks = len(iterable)
                except (TypeError, AttributeError):
                    pass
            # call parent function
            return super().__call__(iterable)
        finally:
            # close tqdm progress bar
            if self.progress_bar is not None:
                self.progress_bar.close()

    __call__.__doc__ = Parallel.__call__.__doc__

    def dispatch_one_batch(self, iterator):
        # start progress_bar, if not started yet.
        if self.progress_bar is None:
            self.progress_bar = tqdm.tqdm(
                desc=self.desc,
                total=self.total_tasks,
                disable=self.disable_progressbar,
                unit="tasks",
                dynamic_ncols=True,
            )
        # call parent function
        return super().dispatch_one_batch(iterator)

    dispatch_one_batch.__doc__ = Parallel.dispatch_one_batch.__doc__

    def print_progress(self):
        """Display the process of the parallel execution using tqdm"""
        # if we finish dispatching, find total_tasks from the number of remaining items
        if self.total_tasks is None and self._original_iterator is None:
            self.total_tasks = self.n_dispatched_tasks
            self.progress_bar.total = self.total_tasks
            self.progress_bar.refresh()
        # update progressbar
        self.progress_bar.update(self.n_completed_tasks - self.progress_bar.n)


def get_mapping_stats(mapping_file: str) -> dict:
    """
    This function takes as input a mapping file containing matches between Amazon items and Wikidata entities, and
    it computes statistics to count the number of matched items. In particular, it computes how many items have been
    matched with title, title+year, title+person, and title+person+year.

    The stats are returned in a dict.

    :param mapping_file: file containing matches between Amazon items and Wikidata entities
    :return: dictionary reporting the stats
    """
    with open(mapping_file) as json_file:
        m = json.load(json_file)

    count = {}
    for k, v in m.items():
        if isinstance(v, dict):
            if v["matching_attributes"] not in count:
                count[v["matching_attributes"]] = 1
            else:
                count[v["matching_attributes"]] += 1

    return count
