from numpy import isin
from utils import preprocess_opts
from kgtk.cli_entry import cli_entry


def kgtk_query(
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
) -> None:
    """
    Executes a KGTK query on the input graph.

    Args:
        input_graph (str): The path to the input graph file.
        output_path (str, optional): The path to the output file. Defaults to "".
        debug (bool, optional): Enable debug mode. Defaults to False.
        graph_cache (str, optional): The path to the graph cache file. Defaults to "".
        match_clause (str, optional): The MATCH clause of the KGTK query. Defaults to "".
        where_clause (str, optional): The WHERE clause of the KGTK query. Defaults to "".
        return_clause (str, optional): The RETURN clause of the KGTK query. Defaults to "".
        index_mode (str, optional): The index mode of the KGTK query. Defaults to "".
        limit (str, optional): The limit of the KGTK query. Defaults to "".
        single_user (bool, optional): Enable single-user mode. Defaults to False.
        read_only (bool, optional): Enable read-only mode. Defaults to False.
    """
    options = [
        ("--debug", debug),
        ("query"),
        ("-i", input_graph),
        ("--graph-cache", graph_cache),
        ("--match", match_clause),
        ("--where", where_clause),
        ("--return", return_clause),
        ("-o", output_path),
        ("--idx", index_mode),
        ("--read-only", read_only),
        ("--single-user", single_user),
        ("--limit", limit),
    ]

    cli_entry(*preprocess_opts(options))
