# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

import os
import logging
import pickle as pkl


def get_logger(name):
    """
    Return a logger object with the specified name.

    Parameters
    ----------
    name : str
        The name of the logger.

    Returns
    -------
    logger_ : logging.Logger
        The logger object.

    """
    logging.basicConfig(format="[%(filename)s:%(lineno)d - %(levelname)s] %(message)s".format(name), level=logging.INFO)
    logger_ = logging.getLogger(name)

    return logger_


def save_pickle(obj, file_name, file_path):
    """
    Save an object to a file.

    Parameters
    ----------
    obj : object
        The object to be saved.
    path : str
        The path to the file where the object will be saved.

    Returns
    -------
    None
    """
    if file_path is None:
        file_path = os.getcwd()

    abs_file_path = os.path.join(file_path, file_name)
    obj_name = type(obj).__name__

    try:
        with open(os.path.join(abs_file_path), "wb") as f:
            pkl.dump(obj, f)
    except Exception as e:
        logger.error(f"Could not save {obj_name} with label {obj.label} object to {abs_file_path}: {e}")

    logger.info(f"Saved {obj_name} with label {obj.label} object to {abs_file_path}")


logger = get_logger(__name__)
