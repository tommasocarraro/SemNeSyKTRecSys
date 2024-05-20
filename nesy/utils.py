from os.path import basename, splitext


def remove_ext(file_path: str) -> str:
    """
    Removes the file extension from a given file path.
    """
    split = splitext(basename(file_path))
    if split[-1] == ".gz":
        return splitext(split[0])[0]
    return split[0]


def compute_graph_extension(compress: bool) -> str:
    """
    Compute the file extension for a graph file based on the compression flag.
    If compress is set to True then the extension is .tsv.gz, otherwise it is .tsv.
    """
    return ".tsv.gz" if compress else ".tsv"
