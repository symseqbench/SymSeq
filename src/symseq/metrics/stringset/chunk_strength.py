# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Associative Chunk Strength (ACS) metrics."""

import numpy as np
from collections import Counter

from symseq.utils.strtools import chunk_ngrams_string_set


def acs_bailey2008(
    train_string_set: list[list[str]], test_string_set: list[list[str]], n_range: tuple[int, int] = (2, 3)
) -> list[float]:
    """
    Compute chunk strength for each test sequence as the average frequency
    of its n-grams in the training set.

    Chunk strength for a chunk g is defined as:
        S(g) = F(g) / (F(g) + E_n)
    where:
        F(g) = frequency of chunk g in training
        E_n  = average frequency of all distinct n-grams in training

    The score for a test sequence is the average chunk strength over all its n-grams.

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
        Chunk strength scores for each test sequence.

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

    strengths = [np.mean(s) if s else 0.0 for s in strengths]
    return strengths


def global_acs_knowlton96(
    train_string_set: list[list[str]], test_string_set: list[list[str]], n_range: tuple[int, int] = (2, 3)
) -> list[float]:
    """
    Compute the global associative chunk strength (GACS) for each string in a test set.

    The chunk strength is defined in terms of the absolute frequency of a chunk
    (the number of times that chunk appeared in the training strings). The chunk
    strength is computed (summed) over all n-grams up to the specified range.

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

    for string_idx, string in enumerate(test_string_set):
        total_ngrams = 0
        chunk_freqs = []

        for n in range(n_range[0], n_range[1] + 1):
            ngrams_train = chunk_ngrams_string_set(train_string_set, n)
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
