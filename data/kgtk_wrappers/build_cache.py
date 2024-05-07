import os
from .query import query


def build_cache(
    input_graph: str,
    graph_cache: str,
    index_mode: str = "",
    debug: bool = False,
):
    query(
        input_graph=input_graph,
        graph_cache=graph_cache,
        output_path=os.devnull,
        index_mode=index_mode,
        debug=debug,
        single_user=True,
        limit="0",
    )
