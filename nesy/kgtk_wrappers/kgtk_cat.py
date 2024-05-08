from kgtk.cli_entry import cli_entry


def kgtk_cat(input_graphs: list[str], output_path: str) -> None:
    """
    Concatenates multiple KGTK graphs into a single output graph file.

    Args:
        input_graphs (list[str]): A list of input graph files to be concatenated.
        output_path (str): The path to the output graph file.
    """
    args = ["kgtk", "cat", "-i", *input_graphs, "-o", output_path]
    cli_entry(*args)
