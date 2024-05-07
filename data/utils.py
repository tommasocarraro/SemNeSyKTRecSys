from os.path import splitext, basename


def remove_empty_strings(args: list[str]):
    return [arg for arg in args if arg]


def remove_ext(file_path: str) -> str:
    split = splitext(basename(file_path))
    if split[-1] == ".gz":
        return splitext(split[0])[0]
    return split[0]


def compute_extension(compress: bool) -> str:
    return ".tsv.gz" if compress else ".tsv"
