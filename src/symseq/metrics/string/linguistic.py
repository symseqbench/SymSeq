# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Linguistic complexity metrics."""

import numpy as np

from symseq.utils.strtools import chunk_ngrams_string


def linguistic_complexity(sequence: list[str], max_w: int | None = None) -> float:
    """
    Compute linguistic complexity across multiple n-gram scales.

    Based on Trifonov (1990), Orlov & Potapov (2004).

    Parameters
    ----------
    sequence : list of str
        Token sequence.
    max_w : int, optional
        Maximum n-gram order. If None, computed as floor(log_|A|(|S|)).

    Returns
    -------
    float
        Linguistic complexity C_ling in (0, 1].

    Notes
    -----
    C_ling = product_{i=1}^W U_i(S)
    where U_i(S) = |V_i(S)| / min(|A|^i, |S| - i + 1)

    Higher values indicate greater vocabulary richness across scales.

    References
    ----------
    Trifonov, E.N. (1990). Making sense of the human genome.
    Orlov, Y.L. & Potapov, V.N. (2004). Complexity: an internet resource
    for analysis of DNA sequence complexity. Nucleic Acids Research, 32.

    Examples
    --------
    >>> linguistic_complexity(['A', 'B', 'C', 'A', 'B'])
    0.XXX
    """
    sequence = np.array(sequence) if not isinstance(sequence, np.ndarray) else sequence
    alphabet = set(sequence)
    alphabet_size = len(alphabet)
    seq_len = len(sequence)

    if seq_len == 0:
        return 0.0

    if max_w is None:
        if alphabet_size > 1:
            max_w = int(np.floor(np.log(seq_len) / np.log(alphabet_size)))
            max_w = max(1, max_w)
        else:
            max_w = 1

    U_product = 1.0

    for i in range(1, max_w + 1):
        all_ngrams, unique_ngrams = chunk_ngrams_string(sequence, n=i)

        if not unique_ngrams:
            break

        vocab_size = len(unique_ngrams)
        max_possible = min(alphabet_size ** i, seq_len - i + 1)

        if max_possible == 0:
            break

        U_i = vocab_size / max_possible
        U_product *= U_i

    return U_product
