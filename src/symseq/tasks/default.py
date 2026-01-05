# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""
targets.py

This module contains functions for generating default outputs for various symbolic sequence tasks.
"""

# from symseq.metrics
from symseq.utils.io import get_logger
from symseq.utils.strtools import chunk_ngrams_string

logger = get_logger(__name__)


def classification(sequence: list[str], skip_eos=True, eos="#") -> tuple[list[str], list[bool]]:
    """
    Create the outputs for sequence classification tasks.

    Parameters
    ----------
    sequence : list[str]
        The sequence to classify.
    skip_eos : bool, optional
        Whether to skip the end of sequence token, by default True.
    eos : str, optional
        The end of sequence token, by default "#".

    Returns
    -------
    tuple[list[str], list[bool]]
        A tuple containing:
        - output: list
            The sequence with target labels appended.
        - accept: list of bool
            Boolean list indicating positions with valid data (not None).
    """
    output = sequence.copy()
    if skip_eos:
        accept = [x != eos for x in output]
    else:
        accept = [True] * len(output)
    return output, accept


def grammaticality_classification(
    sequence: list[str], grammaticality: list[bool], eos="#"
) -> tuple[list[str], list[bool]]:
    """
    Create the outputs for grammaticality (sequence) classification task.

    Parameters
    ----------
    sequence : list of str
        The input sequence of concatenated strings to be classified.
    grammaticality : list of bool
        The grammaticality of each string in the sequence.
    eos : str, optional
        The end-of-string marker, by default "#".

    Returns
    -------
    tuple
        A tuple containing:
        - output: list
            The sequence with target labels appended.
        - accept: list of bool
            Boolean list indicating positions with valid data (not None).

    """
    output = []
    accept = []
    g_iter = iter(grammaticality)

    for i in range(len(sequence)):
        if i + 1 < len(sequence) and sequence[i + 1] == eos:
            accept.append(True)
            output.append(next(g_iter, None))
        else:
            accept.append(False)
            output.append(None)

    return output, accept


def memory_targets(sequence: list[str], n_mem: int) -> tuple[list[str], list[bool]]:
    """
    Create the outputs for sequence memorization tasks.

    Parameters
    ----------
    sequence : list of str
        The input sequence to be memorized.
    n_mem : int
        Number of memory steps.sequence: list[str],t
            The sequence with `None` values inserted for memorization.
        - accept: list of bool
            Boolean list indicating positions with valid data (not None).
    """
    output = sequence[: -(n_mem + 1)]
    while len(output) < len(sequence):
        output.insert(0, None)
    accept = [x is not None for x in output]
    return output, accept


def prediction_targets(sequence: list[str], n_pred: int) -> tuple[list[str], list[bool]]:
    """
    Create the outputs for sequence prediction tasks.

    Parameters
    ----------
    sequence : list of str
        The input sequence for predictions.
    n_pred : int
        Number of prediction steps.

    Returns
    -------
    tuple
        A tuple containing:
        - output: list
            The sequence with predictions appended.
        - accept: list of bool
            Boolean list indicating positions with valid data (not None).
    """
    output = sequence[(n_pred + 1) :]
    while len(output) < len(sequence):
        output.append(None)
    accept = [x is not None for x in output]
    return output, accept


def chunk_targets(
    chunk_seq: list[list[str]], chunk_size: int, max_chunk_memory: int = 0, max_chunk_prediction: int = 0
) -> list:
    """
    Create the outputs for sequence chunking tasks.

    Parameters
    ----------
    chunk_seq : list
        Sequence of chunks.
    n_chunk : int
        Length of each chunk.
    max_chunk_memory : int, optional
        Maximum memory steps for chunks (default is 0).
    max_chunk_prediction : int, optional
        Maximum prediction steps for chunks (default is 0).

    Returns
    -------
    targets : list of dict
        A list of dictionaries, each with keys 'label', 'output', and 'accept'
        describing different chunk targets.
    """
    targets = []
    output = chunk_seq
    for idx in range(chunk_size - 1):
        output.insert(idx, None)
    accept = [x is not None for x in output]

    if chunk_size > 0:
        targets.append({"label": f"{chunk_size}-chunk recognition", "output": output, "accept": accept})
        for n_mem in range(max_chunk_memory):
            output, accept = memory_targets(output, n_mem)
            targets.append({"label": f"{chunk_size}-chunk {n_mem + 1}-memory", "output": output, "accept": accept})
        for n_pred in range(max_chunk_prediction):
            output, accept = prediction_targets(output, n_pred)
            targets.append({"label": f"{chunk_size}-chunk {n_pred + 1}-prediction", "output": output, "accept": accept})
    return targets


def generate_default_outputs(
    input_sequence: list[str],
    max_memory: int = 0,
    max_chunk: int = 0,
    max_prediction: int = 0,
    chunk_memory: bool = False,
    chunk_prediction: bool = False,
    verbose: bool = False,
) -> list:
    """
    Generate default outputs for a variety of symbolic sequence tasks, including
    classification, memory, prediction, and chunk-related tasks.

    Parameters
    ----------
    input_sequence : list
        The input sequence for which to generate outputs.
    max_memory : int, optional
        Maximum number of memory steps to generate (default is 0).
    max_chunk : int, optional
        Maximum chunk length to consider (default is 0).
    max_prediction : int, optional
        Maximum number of prediction steps to generate (default is 0).
    chunk_memory : bool, optional
        Whether to include memory steps within chunks (default is False).
    chunk_prediction : bool, optional
        Whether to include prediction steps within chunks (default is False).
    verbose : bool, optional
        If True, logs detailed information about the generated outputs (default is False).

    Returns
    -------
    targets : list of dict
        A list of dictionaries with task information, where each dictionary
        contains 'label', 'output', and 'accept' keys.
    """
    targets = [{"label": "classification", "output": input_sequence, "accept": [True] * len(input_sequence)}]
    for n_mem in range(max_memory):
        output, accept = memory_targets(input_sequence, n_mem)
        targets.append({"label": f"{n_mem + 1}-step memory", "output": output, "accept": accept})
    for chunk_size in range(2, max_chunk + 1):
        chunks, _ = chunk_ngrams_string(input_sequence, n=chunk_size)
        ch_mem = max_memory if chunk_memory else 0
        ch_pred = max_prediction if chunk_prediction else 0
        targets.extend(chunk_targets(chunks, chunk_size, max_chunk_memory=ch_mem, max_chunk_prediction=ch_pred))
    for n_pred in range(max_prediction):
        output, accept = prediction_targets(input_sequence, n_pred)
        targets.append({"label": f"{n_pred + 1}-step prediction", "output": output, "accept": accept})

    if verbose:
        logger.info(f"Generated {len(targets)} decoder outputs: {[x['label'] for x in targets]}")

    return targets
