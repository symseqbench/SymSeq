# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Distance metrics for string sets."""

import numpy as np
import nltk
from gzip import compress
from tqdm import tqdm
from joblib import Parallel, delayed


def hamming_distance(seq1: list[str], seq2: list[str]) -> int:
    """
    Calculate the Hamming distance between two sequences.

    The Hamming distance is defined as the number of positions at which
    the corresponding symbols are different.

    Parameters
    ----------
    seq1 : list of str
        The first sequence for comparison.
    seq2 : list of str
        The second sequence for comparison.

    Returns
    -------
    int
        The Hamming distance between the two sequences.

    Raises
    ------
    ValueError
        If the sequences have different lengths.

    Notes
    -----
    Properties:
    - Metric: satisfies triangle inequality
    - 0 <= d_H <= n where n is sequence length
    - Only defined for equal-length strings

    Examples
    --------
    >>> hamming_distance(['k','a','r','o','l','i','n'], ['k','a','t','h','r','i','n'])
    3
    """
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of equal length")

    return sum(map(str.__ne__, seq1, seq2))


# TODO add parameters for deletion / insertion costs
def edit_distance(seq1: list[str], seq2: list[str], substitution_cost: int = 1) -> int:
    """
    Calculate the edit distance (Levenshtein distance) between two sequences.

    The edit distance is the minimum number of single-character edits
    (insertions, deletions, substitutions) required to change one
    sequence into the other.

    Parameters
    ----------
    seq1 : list of str
        The first sequence for comparison.
    seq2 : list of str
        The second sequence for comparison.
    substitution_cost : int, default=1
        The cost of substituting a character.

    Returns
    -------
    int
        The edit distance between the two sequences.

    Notes
    -----
    Recursive formulation:
    d_E(i, j) = min(
        d_E(i-1, j) + 1,      # deletion
        d_E(i, j-1) + 1,      # insertion
        d_E(i-1, j-1) + c_ij  # substitution
    )

    Properties:
    - Metric: satisfies triangle inequality
    - |n_i - n_j| <= d_E(S_i, S_j) <= max(n_i, n_j)
    - Generalizes Hamming distance

    Examples
    --------
    >>> edit_distance(['k','i','t','t','e','n'], ['s','i','t','t','i','n','g'])
    3
    """
    return nltk.edit_distance(seq1, seq2, substitution_cost)


def string_similarity(seq1: list[str], seq2: list[str], substitution_cost: int = 1) -> float:
    """
    Calculate the string similarity between two sequences, defined as the edit distance divided by the length of the longest
    sequence.
    """
    return edit_distance(seq1, seq2, substitution_cost) / max(len(seq1), len(seq2))


def average_string_similarity(seq1: list[list[str]], seq2: list[str], substitution_cost: int = 1) -> float:
    """
    Calculate the average string similarity of `seq2` to and a list of sequences `seq1`.
    """
    score = []
    for s1 in seq1:
        score.append(edit_distance(s1, seq2, substitution_cost) / max(len(s1), len(seq2)))
    return np.mean(score)


def normalized_compression_distance(seq1: list[str], seq2: list[str], compressor: str = "gzip") -> float:
    """
    Compute Normalized Compression Distance (NCD) between two sequences.

    Based on Kolmogorov complexity approximation via compression.

    Parameters
    ----------
    seq1, seq2 : list of str
        Input sequences.
    compressor : str, default='gzip'
        Compression method ('gzip' only for now).

    Returns
    -------
    float
        NCD in [0, 1+epsilon], approximately metric.

    Notes
    -----
    NCD(x,y) = [C(xy) - min(C(x), C(y))] / max(C(x), C(y))
    where C(·) is compressed size.

    Properties:
    - 0 <= NCD <= 1 + epsilon (approximately metric)
    - Measures similarity based on shared information
    - Parameter-free and universal

    References
    ----------
    Cilibrasi, R. & Vitányi, P.M.B. (2005). Clustering by compression.
    IEEE Transactions on Information Theory, 51(4), 1523-1545.

    Examples
    --------
    >>> seq1 = ['A', 'B'] * 10
    >>> seq2 = ['A', 'B'] * 10
    >>> ncd = normalized_compression_distance(seq1, seq2)
    >>> assert ncd < 0.1
    """
    if compressor != "gzip":
        raise NotImplementedError("Only gzip compression supported currently")

    str1 = "".join(map(str, seq1))
    str2 = "".join(map(str, seq2))
    str_concat = str1 + str2

    C_x = len(compress(str1.encode("utf-8")))
    C_y = len(compress(str2.encode("utf-8")))
    C_xy = len(compress(str_concat.encode("utf-8")))

    numerator = C_xy - min(C_x, C_y)
    denominator = max(C_x, C_y)

    if denominator == 0:
        return 0.0

    return numerator / denominator


def _compute_pair(i, j, string_set, dist_func, kwargs):
    """Compute single pairwise distance (helper for parallelization)."""
    try:
        return i, j, dist_func(string_set[i], string_set[j], **kwargs)
    except Exception:
        return i, j, np.nan


def pairwise_distances(string_set: list[list[str]], metric: str = "edit", **kwargs) -> np.ndarray:
    """
    Compute pairwise distance matrix for string set.

    Parameters
    ----------
    string_set : list of list of str
        Collection of sequences.
    metric : str, default='edit'
        Distance metric: 'edit', 'hamming', or 'ncd'.
    **kwargs : dict
        Additional arguments for distance function.

    Returns
    -------
    np.ndarray
        T x T distance matrix where T is the number of strings.

    Notes
    -----
    The distance matrix is symmetric with zeros on the diagonal.

    Examples
    --------
    >>> strings = [['A', 'B'], ['A', 'C'], ['B', 'C']]
    >>> D = pairwise_distances(strings, metric='edit')
    >>> D.shape
    (3, 3)
    """
    T = len(string_set)
    D = np.zeros((T, T))

    if metric == "edit":
        dist_func = edit_distance
    elif metric == "hamming":
        dist_func = hamming_distance
    elif metric == "ncd":
        dist_func = normalized_compression_distance
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Calculate total number of pairs
    total_pairs = T * (T - 1) // 2

    with tqdm(total=total_pairs, desc=f"Computing {metric} distances") as pbar:
        for i in range(T):
            for j in range(i + 1, T):
                try:
                    d = dist_func(string_set[i], string_set[j], **kwargs)
                    D[i, j] = d
                    D[j, i] = d
                except ValueError:
                    D[i, j] = np.nan
                    D[j, i] = np.nan
                pbar.update(1)

    return D


def pairwise_distances_parallel(
    string_set: list[list[str]], metric: str = "edit", n_jobs: int = -1, **kwargs
) -> np.ndarray:
    """
    Compute pairwise distance matrix for string set using parallel processing.

    Parameters
    ----------
    string_set : list of list of str
        Collection of sequences.
    metric : str, default='edit'
        Distance metric: 'edit', 'hamming', or 'ncd'.
    n_jobs : int, default=-1
        Number of parallel jobs. -1 means using all processors.
    **kwargs : dict
        Additional arguments for distance function.

    Returns
    -------
    np.ndarray
        T x T distance matrix where T is the number of strings.

    Notes
    -----
    The distance matrix is symmetric with zeros on the diagonal.
    This parallelized version provides significant speedup for large datasets.

    Examples
    --------
    >>> strings = [['A', 'B'], ['A', 'C'], ['B', 'C']]
    >>> D = pairwise_distances_parallel(strings, metric='edit', n_jobs=-1)
    >>> D.shape
    (3, 3)
    """
    T = len(string_set)
    D = np.zeros((T, T))

    if metric == "edit":
        dist_func = edit_distance
    elif metric == "hamming":
        dist_func = hamming_distance
    elif metric == "ncd":
        dist_func = normalized_compression_distance
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Generate all upper triangle pairs
    pairs = [(i, j) for i in range(T) for j in range(i + 1, T)]

    # Parallel computation
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_compute_pair)(i, j, string_set, dist_func, kwargs)
        for i, j in tqdm(pairs, desc=f"Computing {metric} distances (parallel)")
    )

    # Fill matrix
    for i, j, d in results:
        D[i, j] = d
        D[j, i] = d

    return D
