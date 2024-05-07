from utils import remove_empty_strings
from kgtk.cli_entry import cli_entry


def filter(
    input_graph: str, word_separator: str, invert: bool, pattern: str, output_path: str
):
    args = [
        "kgtk",
        "filter",
        "-i",
        input_graph,
        "--word-separator",
        word_separator,
        "--invert" if invert else "",
        "-p",
        pattern,
        "-o",
        output_path,
    ]
    cli_entry(*remove_empty_strings(args))
