from os.path import splitext, basename
from typing import Union


def preprocess_opts(
    opts: list[Union[str, tuple[str, Union[str, bool, list[str]]]]]
) -> list[str]:
    """
    Preprocesses the given options and returns a list of arguments.

    Args:
        opts (list[Union[str, tuple[str, Union[str, bool, list[str]]]]]): The list of options to preprocess.

    Returns:
        list[str]: The list of processed arguments.
    """
    args = ["kgtk"]
    for opt in opts:
        if isinstance(opt, str):
            args.append(opt)
        elif len(opt) == 2:
            option, value = opt
            if isinstance(value, str) and value:
                args.append(option)
                args.append(value)
            elif isinstance(value, bool) and value:
                args.append(option)
            elif isinstance(value, list):
                args.append(option)
                for v in value:
                    args.append(v)
    return args


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
