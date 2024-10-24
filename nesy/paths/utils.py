import subprocess

from loguru import logger


def count_lines(file_path):
    """
    This function efficiently counts the number of lines in a given file.

    :param file_path: path to the file
    :return: number of lines of the given file
    """
    try:
        result = subprocess.run(
            ["wc", "-l", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        output = result.stdout.decode().strip()
        line_count = int(output.split()[0])
        return line_count
    except subprocess.CalledProcessError as e:
        logger.error(f"Error: {e}")
        return -1
