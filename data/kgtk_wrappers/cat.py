from kgtk.cli_entry import cli_entry


def cat(input_graphs: list[str], output_path: str):
    args = ["kgtk", "cat", "-i", *input_graphs, "-o", output_path]
    cli_entry(*args)
