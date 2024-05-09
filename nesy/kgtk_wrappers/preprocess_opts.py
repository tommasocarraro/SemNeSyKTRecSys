from typing import Sequence, Union


def preprocess_opts(
    opts: Sequence[Union[str, tuple[str, Union[str, bool, list[str]]]]]
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
