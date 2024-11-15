import json
from typing import Optional

import tqdm
from joblib import Parallel


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
