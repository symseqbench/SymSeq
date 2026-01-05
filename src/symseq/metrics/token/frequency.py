# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Token frequency metrics."""

from collections import Counter


def token_frequency(sequence: list[str], as_freq: bool = True) -> dict:
    """
    Compute occurrences of each token in sequence, either as raw counts or as frequencies.

    Parameters
    ----------
    sequence : list of str
        Sequence of tokens.
    as_freq : bool, default=True
        Return frequencies (True) or raw counts (False).

    Returns
    -------
    dict
        Token -> frequency/count mapping.

    Examples
    --------
    >>> token_frequency(['A', 'B', 'A', 'C'])
    {'A': 0.5, 'B': 0.25, 'C': 0.25}
    """
    if as_freq:
        return {k: v / len(sequence) for k, v in Counter(sequence).items()}
    else:
        return dict(Counter(sequence))


def most_common_tokens(sequence: list[str], n: int = 10, as_freq: bool = True) -> dict:
    """
    Get n most frequent tokens.

    Parameters
    ----------
    sequence : list of str
        Sequence of tokens.
    n : int, default=10
        Number of top tokens to return.
    as_freq : bool, default=True
        Return frequencies (True) or raw counts (False).

    Returns
    -------
    dict
        Dictionary of n most common tokens with their frequency/count.

    Examples
    --------
    >>> most_common_tokens(['A']*5 + ['B']*3 + ['C'], n=2)
    {'A': 0.556, 'B': 0.333}
    """
    # return most_common(sequence, n=n, as_freq=as_freq)
    ctr = Counter(sequence).most_common(n)
    if as_freq:
        return {k: v / len(sequence) for (k, v) in ctr}
    else:
        return dict(ctr)


def legal_entry(train_string_set: list[list[str]], string: list[str]) -> bool:
    raise NotImplementedError
