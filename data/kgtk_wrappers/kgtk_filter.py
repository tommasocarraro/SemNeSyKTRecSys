from utils import preprocess_opts
from kgtk.cli_entry import cli_entry


def kgtk_filter(
    input_graph: str, word_separator: str, invert: bool, pattern: str, output_path: str
) -> None:
    """
    Filter the input graph using the specified pattern and save the filtered graph to the output path.

    Args:
        input_graph (str): The path to the input graph file.
        word_separator (str): The word separator used in the input graph.
        invert (bool): Whether to invert the filter pattern.
        pattern (str): The filter pattern to apply.
        output_path (str): The path to save the filtered graph.
    """
    options = [
        ("filter"),
        ("-i", input_graph),
        ("--word-separator", word_separator),
        ("--invert", invert),
        ("-p", pattern),
        ("-o", output_path),
    ]
    cli_entry(*preprocess_opts(options))
