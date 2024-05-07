from kgtk.cli_entry import cli_entry


def add_id(input_graph: str, output_path: str, id_style: str = "wikidata"):
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
