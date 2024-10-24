import os

from .kgtk_query import kgtk_query


def kgtk_build_cache(
    input_graph: str,
    graph_cache: str,
    index_mode: str = "",
    debug: bool = False,
) -> None:
    """
    Builds a cache for the input graph using KGTK.

    Args:
        input_graph (str): The path to the input graph file.
        graph_cache (str): The path to the output cache file.
        index_mode (str, optional): The index mode to use. Defaults to "".
        debug (bool, optional): Whether to enable debug mode. Defaults to False.
    """
    kgtk_query(
        input_graph=input_graph,
        graph_cache=graph_cache,
        output_path=os.devnull,
        index_mode=index_mode,
        debug=debug,
        single_user=True,
        limit="0",
    )
