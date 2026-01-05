# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Permutation entropy and ordinal pattern analysis."""

import numpy as np
from itertools import permutations

try:
    from antropy import perm_entropy as antropy_perm_entropy
    ANTROPY_AVAILABLE = True
except ImportError:
    ANTROPY_AVAILABLE = False


def permutation_entropy(
    sequence: list[str] | np.ndarray,
    order: int = 3,
    delay: int = 1,
    normalize: bool = False
) -> float:
    """
    Compute permutation entropy (Bandt & Pompe, 2002).

    Uses antropy library if available (10-50x faster), otherwise falls back
    to custom implementation.

    Parameters
    ----------
    sequence : list or np.ndarray
        Symbolic or numeric sequence.
    order : int, default=3
        Embedding dimension (pattern length).
    delay : int, default=1
        Time delay for embedding.
    normalize : bool, default=False
        Normalize by log2(order!).

    Returns
    -------
    float
        Permutation entropy H_perm in [0, log2(order!)].
        If normalized, H_perm in [0, 1].

    Notes
    -----
    Permutation entropy quantifies complexity based on ordinal patterns.
    Robust to noise and non-stationarity.

    For symbolic sequences, tokens are converted to numeric values via hashing.

    References
    ----------
    Bandt, C. & Pompe, B. (2002). Permutation entropy: a natural complexity
    measure for time series. Physical Review Letters, 88(17), 174102.

    Examples
    --------
    >>> seq = np.random.randn(1000)
    >>> H = permutation_entropy(seq, order=3)
    """
    if isinstance(sequence, list):
        if len(sequence) > 0 and isinstance(sequence[0], str):
            sequence = np.array([hash(s) % 10000 for s in sequence], dtype=float)
        else:
            sequence = np.asarray(sequence, dtype=float)
    else:
        sequence = np.asarray(sequence, dtype=float)

    if ANTROPY_AVAILABLE:
        return antropy_perm_entropy(sequence, order=order, delay=delay, normalize=normalize)
    else:
        return _custom_permutation_entropy(sequence, order, delay, normalize)


def _custom_permutation_entropy(sequence, order, delay, normalize):
    """Custom permutation entropy implementation (fallback)."""
    n = len(sequence)
    permutation_counts = {}

    all_perms = list(permutations(range(order)))
    for perm in all_perms:
        permutation_counts[perm] = 0

    for i in range(n - delay * (order - 1)):
        indices = [i + delay * j for j in range(order)]
        values = [sequence[idx] for idx in indices]
        ordinal_pattern = tuple(np.argsort(values))
        permutation_counts[ordinal_pattern] += 1

    total_patterns = sum(permutation_counts.values())
    if total_patterns == 0:
        return 0.0

    H = 0.0
    for count in permutation_counts.values():
        if count > 0:
            p = count / total_patterns
            H -= p * np.log2(p)

    if normalize:
        import math
        max_H = np.log2(math.factorial(order))
        H = H / max_H if max_H > 0 else 0.0

    return H
