# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

from collections import Counter
import numpy as np
import copy
import pandas as pd

# local imports
from ..utils.io import get_logger
from symseq.core.state import State

logger = get_logger(__name__)

# TODO decide when to use sequence and when to use strings?!


def string_set_length(string_set: list[list[str]]) -> list[int]:
    # TODO string or sequence?
    """
    Compute the length of all strings in the string_set.

    Parameters
    ----------
    string_set : list of list of str
        List of strings to compute lengths for. If None, uses instance's string set.

    Returns
    -------
    list of int
        List with lengths of each string in the string set. Returns empty list if string_set is empty.
    """
    if len(string_set) == 0:
        logger.warning("Provided string set is empty, returning empty list.")
        return []

    return [len(string) for string in string_set]


def sequence_to_strings(sequence: list[str], eos: str) -> list[list[str]]:
    """
    Splits a sequence of symbols into a list of sub-sequences (lists) based on a given separator.

    Parameters:
    -----------
    sequence : list[str]
        The sequence of symbols to split.
    eos : str
        The value in the list used as the separator between strings (subsequences).

    Returns:
    --------
    list of lists
        A list containing sub-lists, split at the points where the separator occurs.
    """
    idxs_sep = [-1] + np.where(np.array(sequence) == eos)[0].tolist() + [len(sequence)]
    string_set = []

    # Use the separator positions to slice the list
    for i in range(len(idxs_sep) - 1):
        if idxs_sep[i + 1] - idxs_sep[i] > 1:
            string_set.append(sequence[idxs_sep[i] + 1 : idxs_sep[i + 1]])

    return string_set


# TODO should EOS be appended after the last string?
def concatenate_string_set(string_set: list[list[str]], eos: str = None) -> list:
    """
    Concatenates a string set into a list of symbols (single sequence), separating them with a specified eos marker.

    Parameters
    ----------
    string_set : list of list of str
        List of strings to concatenate.
    eos : str, optional
        String symbol used to separate different strings. Default is empty string.

    Returns
    -------
    list of str
        Concatenated list of symbols with separator inserted between strings.

    Examples
    --------
    >>> concatenate_string_set([["A", "B", "C"], ["D", "E", "F"]])
    ["A", "B", "C", "D", "E", "F"]
    >>> concatenate_string_set([["A", "B", "C"], ["D", "E", "F"]], eos="#")
    ["A", "B", "C", "#", "D", "E", "F"]
    """
    str_set = copy.deepcopy(string_set)
    if eos is not None:
        [n.insert(0, eos) for idx, n in enumerate(list(str_set)) if idx != 0]
    sequence = np.concatenate(list(str_set)).tolist()
    return sequence


def clean_string_set(
    string_set: list[list[str]],
    eos: str | None = None,
    as_states: bool = False,
) -> list[list[str]]:
    """
    Cleans the string set by removing the terminal symbol (EOS) and converting the strings from (indexed) states to
    symbols if necessary and appropriate. This function creates a new string set and does not modify the original one.

    Note:
        - Assumes that each EOS only occurs at most once in each string

    Parameters
    ----------
    string_set : list of list of str
        List of strings to clean.
    eos : str, optional
        Symbol to use as the EOS marker. If None, EOS will not be removed. Defaults to None.
    as_states : bool
        If True, the string will be returned as a list of states (e.g., A, B3) instead of symbols (e.g., A, B), if the
        provided string set contains indexed states and not symbols. Defaults to False.

    Returns
    -------
    list of list of str
        Cleaned string set as new object.
    """
    new_string_set = []

    for string in string_set:
        new_string = copy.deepcopy(string)

        if eos is not None and eos in string:
            assert len(np.where(np.array(string) == eos)[0]) == 1, f"EOS {eos} occurs > 1?"
            new_string.remove(eos)

        if not as_states:
            new_string = string_as_symbols(string)

        new_string_set.append(new_string)

    return new_string_set


def string_as_symbols(string: list[str] | list[State]) -> list[str]:
    """
    Convert a string (list of states/symbols) to a list of symbols by removing the indices.

    Parameters
    ----------
    string : list of str or list of State
        List of states to convert to symbols.

    Returns
    -------
    list of str
        List of symbols corresponding to the states.
    """
    if len(string) == 0:
        return []

    symbols = []
    for s in string:
        if isinstance(s, State):
            symbols.append(s.symbol)
        elif isinstance(s, str):
            symbols.append((State.from_string(s)).symbol)
        else:
            raise ValueError(f"Expected State or str, got {type(s)}")

    return symbols


def chunk_ngrams_string_set(string_set: list[list[str]], n: int) -> dict:
    """
    Counts the frequency of n-grams in a string set (pooled across all strings).

    Parameters
    ----------
    string_set : list of list of str
        List of strings to count n-grams from.
    n : int
        The order parameter specifying the length of each n-gram.

    Returns
    -------
    dict
        Dictionary with n-grams as keys and their frequency as values.

    Examples
    --------
    """
    ngrams = []
    for string in string_set:
        ngrams.extend(tuple(string[i : i + n]) for i in range(len(string) - n + 1))

    ngram_freqs = Counter(ngrams)

    return dict(ngram_freqs)


def chunk_ngrams_string(sequence: list[str], n: int) -> tuple[list[list[str]], list[str]]:
    """
    Generates n-grams from a sequence and determines their unique occurrences.

    This function creates all possible contiguous subsequences of length 'n' (n-grams)
    from the input `seq`, then filters out and returns only the unique n-grams.

    Parameters
    ----------
    sequence : list
        The full symbolic sequence from which n-grams are generated.
    n : int
        The order parameter specifying the length of each n-gram.

    Returns
    -------
    all_ngrams : list of list of str
        List of all n-grams from the sequence.
    unique_ngrams : list of list
        List of unique n-grams.

    Examples
    --------
    >>> chunk_ngrams(['a', 'b', 'c', 'd'], 2)
    (['ab', 'bc', 'cd'], ['ab', 'bc', 'cd'])

    >>> chunk_ngrams('abc', 2)
    (['ab', 'bc'], ['ab', 'bc'])
    """
    seq_list = list(sequence)
    seq_size = len(seq_list)
    if seq_size < n:
        return [], []

    all_ngrams = ["".join(seq_list[ii : ii + n]) for ii in range(seq_size - n + 1)]
    unique_ngrams = list(set(all_ngrams))

    return all_ngrams, unique_ngrams


# TODO double check for correctness; write test
def chunk_transitions(
    sequence: list[str], chunk_len: int, verbose: bool = True, as_dataframe: bool = False
) -> np.ndarray | pd.DataFrame:
    """
    Determine the transition matrix for n-gram sequences.

    Parameters
    ----------
    seq: [list or generator]
            Full symbolic sequence (should be as long as possible, particularly for large n)
    chunk_len: int
            Order parameter (chunk "length")
    verbose: bool
            Show transition table
    as_dataframe: bool
            If True, return the transition table as a pandas DataFrame, otherwise return as a numpy array.

    Returns
    -------
    np.ndarray
        Transition matrix.

    """
    all_ngrams, unique_ngrams = chunk_ngrams_string(sequence, chunk_len)
    assert (
        len(np.unique([len(x) for x in unique_ngrams])) == 1
    ), f"All unique n-grams must have the same length n={chunk_len}!"
    assert len(np.unique([len(x) for x in all_ngrams])) == 1, f"All n-grams must have the same length n={chunk_len}!"

    unique_ngrams = np.array(unique_ngrams)
    all_ngrams = np.array(all_ngrams)

    Mfast_lin = np.zeros((len(unique_ngrams), len(unique_ngrams)))

    # precalculate indices for each unique n_gram
    un_ngram_dict_to_idx = {ngram: i for i, ngram in enumerate(unique_ngrams)}
    # O(len(all_ngrams)) - linear
    for idx, src_ngram in enumerate(all_ngrams[:-1]):
        tgt_ngram = all_ngrams[idx + 1]
        j = un_ngram_dict_to_idx[tgt_ngram]
        i = un_ngram_dict_to_idx[src_ngram]
        Mfast_lin[i, j] = 1.0

    # # Runtime O(n^2)
    # for ii, ngram_i in enumerate(un_ngrams):
    # 	for jj, ngram_j in enumerate(un_ngrams):
    # 		# previously
    # 		# M[ii, jj] = float(any(all_ngrams[np.where(all_ngrams == ngram_i)[0][:-1] + 1] == ngram_j))
    #
    # 		# current
    # 		# check for any transitions from ngram_i -> ngram_j, i.e., if they occur with a shift of 1 in all_ngrams.
    # 		M[ii, jj] = float(any(all_ngrams[np.where(all_ngrams[:-1] == ngram_i)[0] + 1] == ngram_j))
    # assert np.array_equal(M, Mfast_lin)

    df = pd.DataFrame(Mfast_lin, columns=unique_ngrams, index=unique_ngrams)

    if verbose:
        print(df)

    if as_dataframe:
        return df
    else:
        return Mfast_lin
