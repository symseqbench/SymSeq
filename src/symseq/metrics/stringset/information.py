# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Information-theoretic metrics for string sets."""

import numpy as np
import warnings
from collections import Counter
from tqdm import tqdm
from joblib import Parallel, delayed

from symseq.metrics.string import entropy


def mutual_information_strings(
    seq1: list[str],
    seq2: list[str],
    method: str = 'joint'
) -> float:
    """
    Compute mutual information between two strings.

    Parameters
    ----------
    seq1, seq2 : list of str
        Input sequences.
    method : str, default='joint'
        Method: 'joint' (concatenate) or 'aligned' (position-wise).

    Returns
    -------
    float
        I(S1; S2) = H(S1) + H(S2) - H(S1, S2)

    Notes
    -----
    For 'joint' method, computes entropy of concatenated sequence.
    For 'aligned', requires equal-length sequences.

    Examples
    --------
    >>> seq1 = ['A', 'B', 'A']
    >>> seq2 = ['A', 'B', 'A']
    >>> MI = mutual_information_strings(seq1, seq2)
    """
    if method == 'joint':
        H_x = entropy(seq1)
        H_y = entropy(seq2)
        H_xy = entropy(seq1 + seq2)
    elif method == 'aligned':
        if len(seq1) != len(seq2):
            raise ValueError("Aligned method requires equal-length sequences")
        H_x = entropy(seq1)
        H_y = entropy(seq2)
        paired = list(zip(seq1, seq2))
        H_xy = entropy(paired)
    else:
        raise ValueError(f"Unknown method: {method}")

    MI = H_x + H_y - H_xy
    return max(0.0, MI)


def normalized_mutual_information(
    seq1: list[str],
    seq2: list[str],
    method: str = 'joint'
) -> float:
    """
    Normalized mutual information NMI in [0, 1].

    NMI = 2*I(X;Y) / (H(X) + H(Y))

    Parameters
    ----------
    seq1, seq2 : list of str
        Input sequences.
    method : str, default='joint'
        Method: 'joint' or 'aligned'.

    Returns
    -------
    float
        Normalized mutual information in [0, 1].

    Examples
    --------
    >>> seq1 = ['A', 'B', 'A']
    >>> seq2 = ['A', 'B', 'A']
    >>> NMI = normalized_mutual_information(seq1, seq2)
    """
    MI = mutual_information_strings(seq1, seq2, method=method)
    H_x = entropy(seq1)
    H_y = entropy(seq2)

    denom = H_x + H_y
    if denom == 0:
        return 0.0

    nmi = 2 * MI / denom
    # Clip to [0, 1] to handle numerical precision issues
    return np.clip(nmi, 0.0, 1.0)


def _compute_mi_pair(i, j, string_set, mi_func, method, kwargs):
    """Compute single pairwise MI (helper for parallelization)."""
    try:
        return i, j, mi_func(string_set[i], string_set[j], method=method, **kwargs)
    except Exception:
        return i, j, np.nan


def pairwise_mutual_information(
    string_set: list[list[str]],
    metric: str = 'mi',
    method: str = 'joint',
    **kwargs
) -> np.ndarray:
    """
    Compute pairwise mutual information matrix for string set.

    Parameters
    ----------
    string_set : list of list of str
        Collection of sequences.
    metric : str, default='mi'
        Information metric: 'mi' (mutual information) or 'nmi' (normalized MI).
    method : str, default='joint'
        Method for MI computation: 'joint' or 'aligned'.
    **kwargs : dict
        Additional arguments for MI function.

    Returns
    -------
    np.ndarray
        T x T mutual information matrix where T is the number of strings.

    Notes
    -----
    The MI matrix is symmetric with maximum values on the diagonal.
    For 'aligned' method, sequences must have equal length.

    Examples
    --------
    >>> strings = [['A', 'B'], ['A', 'C'], ['B', 'C']]
    >>> MI = pairwise_mutual_information(strings, metric='mi')
    >>> MI.shape
    (3, 3)
    """
    T = len(string_set)
    MI_matrix = np.zeros((T, T))

    if metric == 'mi':
        mi_func = mutual_information_strings
    elif metric == 'nmi':
        mi_func = normalized_mutual_information
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'mi' or 'nmi'.")

    # Calculate total number of pairs (including diagonal)
    total_pairs = T * (T + 1) // 2
    
    with tqdm(total=total_pairs, desc=f"Computing {metric.upper()}") as pbar:
        for i in range(T):
            for j in range(i, T):
                try:
                    mi_val = mi_func(string_set[i], string_set[j], method=method, **kwargs)
                    MI_matrix[i, j] = mi_val
                    MI_matrix[j, i] = mi_val
                except ValueError:
                    # Handle cases where aligned method fails due to length mismatch
                    MI_matrix[i, j] = np.nan
                    MI_matrix[j, i] = np.nan
                pbar.update(1)

    return MI_matrix


def pairwise_mutual_information_parallel(
    string_set: list[list[str]],
    metric: str = 'mi',
    method: str = 'joint',
    n_jobs: int = -1,
    **kwargs
) -> np.ndarray:
    """
    Compute pairwise mutual information matrix for string set using parallel processing.

    Parameters
    ----------
    string_set : list of list of str
        Collection of sequences.
    metric : str, default='mi'
        Information metric: 'mi' (mutual information) or 'nmi' (normalized MI).
    method : str, default='joint'
        Method for MI computation: 'joint' or 'aligned'.
    n_jobs : int, default=-1
        Number of parallel jobs. -1 means using all processors.
    **kwargs : dict
        Additional arguments for MI function.

    Returns
    -------
    np.ndarray
        T x T mutual information matrix where T is the number of strings.

    Notes
    -----
    The MI matrix is symmetric with maximum values on the diagonal.
    For 'aligned' method, sequences must have equal length.
    This parallelized version provides significant speedup for large datasets.

    Examples
    --------
    >>> strings = [['A', 'B'], ['A', 'C'], ['B', 'C']]
    >>> MI = pairwise_mutual_information_parallel(strings, metric='mi', n_jobs=-1)
    >>> MI.shape
    (3, 3)
    """
    T = len(string_set)
    MI_matrix = np.zeros((T, T))

    if metric == 'mi':
        mi_func = mutual_information_strings
    elif metric == 'nmi':
        mi_func = normalized_mutual_information
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'mi' or 'nmi'.")

    # Generate all pairs including diagonal
    pairs = [(i, j) for i in range(T) for j in range(i, T)]
    
    # Parallel computation
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_compute_mi_pair)(i, j, string_set, mi_func, method, kwargs)
        for i, j in tqdm(pairs, desc=f"Computing {metric.upper()} (parallel)")
    )
    
    # Fill matrix
    for i, j, mi_val in results:
        MI_matrix[i, j] = mi_val
        MI_matrix[j, i] = mi_val
    
    return MI_matrix


def string_set_entropy(
    string_set: list[list[str]],
    use_frequencies: bool = False
) -> float:
    """
    Compute entropy of string-set distribution.

    Parameters
    ----------
    string_set : list of list of str
        Collection of sequences.
    use_frequencies : bool, default=False
        If True, assume strings can repeat and compute empirical distribution.
        If False, assume uniform distribution (1/T for each unique string).

    Returns
    -------
    float
        H(S) = -sum(P(S) * log2(P(S)))

    Notes
    -----
    Requires sufficient samples for reliable estimation (T > 20 recommended).

    Examples
    --------
    >>> strings = [['A', 'B'], ['A', 'B'], ['C', 'D']]
    >>> H = string_set_entropy(strings, use_frequencies=True)
    """
    if len(string_set) < 20:
        warnings.warn(
            "String set too small (T < 20) for reliable entropy estimation",
            UserWarning
        )

    if use_frequencies:
        string_tuples = [tuple(s) for s in string_set]
        counts = Counter(string_tuples)
        T = len(string_set)
        probs = np.array(list(counts.values())) / T
    else:
        unique_strings = set(tuple(s) for s in string_set)
        T_unique = len(unique_strings)
        probs = np.full(T_unique, 1.0 / T_unique)

    H = -np.sum(probs * np.log2(probs + 1e-12))
    return H
