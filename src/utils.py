import os
import random
from pathlib import Path

import multivolumefile
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
            # splitting file the extension
            extension = compressed_file_path.suffix
            if compressed_file_path.suffix == ".001":
                filename = Path(compressed_file_path.stem).stem
            else:
                filename = compressed_file_path.stem
            output_path = dirname / filename
            if output_path.is_file():
                return output_path

            if extension == ".7z":
                logger.debug(f"Decompressing {compressed_file_path}")
                with py7zr.SevenZipFile(compressed_file_path, mode="r") as archive:
                    archive.extractall(path=compressed_file_path.parent)
            elif extension == ".001":
                non_part_file_path = compressed_file_path.parent / compressed_file_path.stem
                logger.debug(f"Decompressing {non_part_file_path}")
                with multivolumefile.open(non_part_file_path, mode="rb") as target_archive:
                    with py7zr.SevenZipFile(target_archive, mode="r") as archive:  # type: ignore
                        archive.extractall(path=compressed_file_path.parent)
            else:
                logger.error("Provided file is not a .7z or .7z.00x archive")
                exit(1)

            return output_path

        else:
            logger.error(f"Error. You can only decompress a file. Instead I got: {compressed_file_path}")
            exit(1)

    else:
        logger.error(f"Trying to decompress a file which does not exist: {compressed_file_path}")
        exit(1)
