# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

import logging
import os
import pickle as pkl


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger object with the specified name.

    Parameters
    ----------
    name : str
        The name of the logger.

    Returns
    -------
    logging.Logger
        The logger object.
    """
    return logging.getLogger(name)


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
