import os
import random
from pathlib import Path

import numpy as np
import py7zr
import torch
from loguru import logger


def set_seed(seed: int):
    """
    It sets the seed for the reproducibility of the experiments.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def decompress_7z(compressed_file_path: Path):
    """
    It decompresses a given compressed file.

    :param compressed_file_path: path to the compressed file
    :return: the path without the compression extension
    """
    # check if file exists
    if compressed_file_path.exists():
        # check if path is indeed pointing to a file
        if compressed_file_path.is_file():
            dirname = compressed_file_path.parent
            if compressed_file_path.suffix == ".001":
                filename = Path(compressed_file_path.stem).stem
            else:
                filename = compressed_file_path.stem
            # splitting file the extension
            extension = compressed_file_path.suffix
            output_path = dirname / filename
            if extension == ".7z":
                if not output_path.exists() or not output_path.is_file():
                    logger.debug(f"Decompressing {compressed_file_path}")
                    with py7zr.SevenZipFile(compressed_file_path, mode="r") as archive:
                        archive.extractall(path=compressed_file_path.parent)
            return output_path

        else:
            logger.error(f"Error. You can only decompress a file. Instead I got: {compressed_file_path}")
            exit(1)

    else:
        logger.error(f"Trying to decompress a file which does not exist: {compressed_file_path}")
        exit(1)
