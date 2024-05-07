from utils import remove_empty_strings
from kgtk.cli_entry import cli_entry


def query(
    input_graph: str,
    output_path: str = "",
    debug: bool = False,
    graph_cache: str = "",
    match_clause: str = "",
    where_clause: str = "",
    return_clause: str = "",
    index_mode: str = "",
    limit: str = "",
    single_user: bool = False,
    read_only: bool = False,
):
    args = [
        "kgtk",
        "--debug" if debug else "",
        "query",
        "-i",
        input_graph,
        "--graph-cache" if graph_cache else "",
        graph_cache if graph_cache else "",
        "--match" if match_clause else "",
        match_clause if match_clause else "",
        "--where" if where_clause else "",
        where_clause if where_clause else "",
        "--return" if return_clause else "",
        return_clause if return_clause else "",
        "-o" if output_path else "",
        output_path if output_path else "",
        "--idx" if index_mode else "",
        index_mode if index_mode else "",
        "--read-only" if read_only else "",
        "--single-user" if single_user else "",
        "--limit" if limit else "",
        limit if limit else "",
    ]
    cli_entry(*remove_empty_strings(args))
