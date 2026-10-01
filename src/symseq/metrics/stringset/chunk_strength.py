# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Associative Chunk Strength (ACS) metrics."""

from collections import Counter

import numpy as np

from symseq.utils.strtools import chunk_ngrams_string_set


def acs_bailey2008(
    train_string_set: list[list[str]], test_string_set: list[list[str]], n_range: tuple[int, int] = (2, 3)
) -> list[float]:
    """
    Compute the associative chunk strength of each test sequence, following AGL StimSelect.

    The strength of a chunk (n-gram) g is a saturating function of its training frequency, bounded
    in [0, 1):

        S(g) = F(g) / (F(g) + E_n)

    where:
        F(g) = frequency of chunk g in the training set (0 if unseen)
        E_n  = expected chunk frequency for chunks of size n, i.e. the mean frequency over all
               distinct n-grams occurring in training

    A chunk of average frequency has strength 0.5 and S(g) -> 1 as F(g) >> E_n. The score of a test
    sequence is the mean of S(g) over all of its chunks of every size in `n_range`.

    Parameters
    ----------
    train_string_set : list of list of str
        List of training sequences.
    test_string_set : list of list of str
        List of test sequences.
    n_range : tuple of int, default=(2, 3)
        Range of n-gram sizes to consider (inclusive).

    Returns
    -------
    list of float
        Chunk strength scores in [0, 1) for each test sequence. Sequences shorter than every n in
        `n_range` have no chunks and score 0.0.

    Notes
    -----
    This is not the raw-frequency measure of Knowlton & Squire (1996), which is implemented by
    `global_acs_knowlton96` and whose values are unbounded and grow with the size of the training
    set. The StimSelect measure is normalized by E_n, so it is independent of how often the same
    training strings are repeated and of the number of training strings (Bailey & Pothos, 2008). The
    two are not interchangeable in absolute value, although they typically correlate highly.

    References
    ----------
    Bailey, T. M., & Pothos, E. M. (2008). AGL StimSelect: Software for
    automated selection of stimuli for artificial grammar learning.
    Behavior Research Methods, 40(1), 164-176.
    """
    strengths = [[] for _ in range(len(test_string_set))]

    for n in range(n_range[0], n_range[1] + 1):
        training_ngrams = []
        for string in train_string_set:
            training_ngrams.extend(tuple(string[i : i + n]) for i in range(len(string) - n + 1))

        ngram_freqs = Counter(training_ngrams)

        total_freq = sum(ngram_freqs.values())
        num_unique = len(ngram_freqs)
        E_n = total_freq / num_unique if num_unique > 0 else 0

        for string_idx, string in enumerate(test_string_set):
            ngrams = [tuple(string[i : i + n]) for i in range(len(string) - n + 1)]
            if not ngrams:
                continue

            chunk_scores = []
            for ng in ngrams:
                F = ngram_freqs.get(ng, 0)
                cs = F / (F + E_n) if E_n > 0 else 0.0
                chunk_scores.append(cs)

            strengths[string_idx].extend(chunk_scores)

    return [float(np.mean(chunk_scores)) if chunk_scores else 0.0 for chunk_scores in strengths]


def global_acs_knowlton96(
    train_string_set: list[list[str]], test_string_set: list[list[str]], n_range: tuple[int, int] = (2, 3)
) -> list[float]:
    """
    Compute the global associative chunk strength (GACS) for each string in a test set.

    The chunk strength is defined in terms of the absolute frequency of a chunk
    (the number of times that chunk appeared in the training strings). The score of a test
    string is the mean training frequency of all its n-grams (over every n in `n_range`).
    Values are unbounded and scale with the size of the training set; see `acs_bailey2008`
    for the normalized variant.

    Parameters
    ----------
    train_string_set : list of list of str
        List of training sequences.
    test_string_set : list of list of str
        List of test sequences.
    n_range : tuple of int, default=(2, 3)
        Range of n-gram sizes to consider (inclusive).

    Returns
    -------
    list of float
        Global associative chunk strengths for each test sequence.

    References
    ----------
    Knowlton, B. J., & Squire, L. R. (1996). Artificial grammar learning
    depends on implicit acquisition of both abstract and exemplar-specific
    information. Journal of Experimental Psychology: Learning, Memory, and
    Cognition, 22(1), 169.
    """
    strengths = []
    train_ngrams = {n: chunk_ngrams_string_set(train_string_set, n) for n in range(n_range[0], n_range[1] + 1)}

    for string in test_string_set:
        total_ngrams = 0
        chunk_freqs = []

        for n, ngrams_train in train_ngrams.items():
            ngrams_string = [tuple(string[i : i + n]) for i in range(len(string) - n + 1)]
            total_ngrams += len(ngrams_string)
            ngram_freq = [ngrams_train.get(chunk, 0) for chunk in ngrams_string]
            chunk_freqs.extend(ngram_freq)

        global_score = sum(chunk_freqs) / total_ngrams if total_ngrams > 0 else 0.0
        strengths.append(global_score)

    return strengths


def anchor_acs_knowlton96(
    train_string_set: list[list[str]], test_string_set: list[list[str]], n_range: tuple[int, int] = (2, 3)
) -> list[float]:
    """
    Compute the anchor chunk strength (ACS) for each string in a test set.

    ACS = average of how often the test string's first n-gram occurred as the
    first n-gram of a training string, and how often its last n-gram occurred
    as the last n-gram of a training string.

    Parameters
    ----------
    train_string_set : list of list of str
        List of training sequences.
    test_string_set : list of list of str
        List of test sequences.
    n_range : tuple of int, default=(2, 3)
        Range of n-gram sizes to consider (inclusive).

    Returns
    -------
    list of float
        Anchor chunk strengths for each test sequence.

    References
    ----------
    Knowlton, B. J., & Squire, L. R. (1996). Artificial grammar learning
    depends on implicit acquisition of both abstract and exemplar-specific
    information. Journal of Experimental Psychology: Learning, Memory, and
    Cognition, 22(1), 169.
    """
    first_ngram_counts = {n: Counter() for n in range(n_range[0], n_range[1] + 1)}
    last_ngram_counts = {n: Counter() for n in range(n_range[0], n_range[1] + 1)}

    for s in train_string_set:
        for n in range(n_range[0], n_range[1] + 1):
            if len(s) >= n:
                first_ngram_counts[n][tuple(s[:n])] += 1
                last_ngram_counts[n][tuple(s[-n:])] += 1

    total_acs = []

    for s in test_string_set:
        acs_sum = 0
        count = 0
        for n in range(n_range[0], n_range[1] + 1):
            if len(s) >= n:
                first_ng = tuple(s[:n])
                last_ng = tuple(s[-n:])
                score = first_ngram_counts[n].get(first_ng, 0) + last_ngram_counts[n].get(last_ng, 0)
                acs_sum += score / 2
                count += 1

        total_acs.append(acs_sum / count if count > 0 else 0.0)

    return total_acs


# def mean_global_acs_knowlton96(
#     train_string_set: list[list[str]], test_string_set: list[list[str]], n_range: tuple[int, int] = (2, 3)
# ):
#     """
#     Compute the mean associative chunk strength (GACS) for each string in a test set, averaged over all strings and
#     n-grams within the specified range.
#     """
#     gacs_list = global_acs_knowlton96(train_string_set, test_string_set, n_range)
#     return np.mean(gacs_list)
