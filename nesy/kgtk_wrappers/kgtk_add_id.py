from kgtk.cli_entry import cli_entry


def kgtk_add_id(input_graph: str, output_path: str, id_style: str = "wikidata") -> None:
    """
    Add IDs to the input graph using the specified ID style.

    Args:
        input_graph (str): The path to the input graph file.
        output_path (str): The path to the output file where the graph with added ID will be saved.
        id_style (str, optional): The style of the ID to be added. Defaults to "wikidata".
    """
    args = [
        "kgtk",
        "add-id",
        "-i",
        input_graph,
        "-o",
        output_path,
        "--id-style",
        id_style,
    ]
    cli_entry(*args)
