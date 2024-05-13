from os.path import basename, splitext
import subprocess


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


def count_lines(file_path):
    """
    This function efficiently counts the number of lines in a given file.

    :param file_path: path to the file
    :return: number of lines of the given file
    """
    try:
        result = subprocess.run(['wc', '-l', file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                check=True)
        output = result.stdout.decode().strip()
        line_count = int(output.split()[0])
        return line_count
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return -1
