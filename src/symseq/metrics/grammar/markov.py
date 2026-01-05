# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Markov model analysis and order selection."""

import numpy as np
from collections import defaultdict, Counter
import warnings


def markov_order_selection(
    sequence: list[str],
    max_order: int = 5,
    criterion: str = 'BIC'
) -> dict:
    """
    Select optimal Markov order using information criteria.

    Parameters
    ----------
    sequence : list of str
        Symbolic sequence.
    max_order : int, default=5
        Maximum order to test.
    criterion : str, default='BIC'
        Information criterion: 'BIC' or 'AIC'.

    Returns
    -------
    dict
        Results with keys:
        - 'optimal_order': Selected order
        - 'scores': IC scores for each order
        - 'log_likelihoods': Log-likelihoods for each order
        - 'transition_matrix': Transition matrix for optimal order

    Notes
    -----
    BIC = -2*log(L) + k*log(n)
    AIC = -2*log(L) + 2*k

    where L is likelihood, k is number of parameters, n is sample size.

    Lower IC values indicate better models.

    References
    ----------
    Schwarz, G. (1978). Estimating the dimension of a model.
    Annals of Statistics, 6(2), 461-464.

    Examples
    --------
    >>> seq = ['A', 'B'] * 100
    >>> result = markov_order_selection(seq, max_order=3)
    >>> result['optimal_order']
    1
    """
    n = len(sequence)
    alphabet = list(set(sequence))
    alphabet_size = len(alphabet)

    scores = []
    log_likelihoods = []

    for order in range(max_order + 1):
        if order == 0:
            counts = Counter(sequence)
            total = sum(counts.values())
            log_L = sum(count * np.log(count / total) for count in counts.values())
            k = alphabet_size - 1
        else:
            if n <= order:
                scores.append(np.inf)
                log_likelihoods.append(-np.inf)
                continue

            contexts = defaultdict(lambda: defaultdict(int))
            for i in range(order, n):
                context = tuple(sequence[i - order:i])
                next_symbol = sequence[i]
                contexts[context][next_symbol] += 1

            log_L = 0.0
            # Count number of observed contexts (for reporting)
            num_observed_contexts = len(contexts)
            
            # Calculate log-likelihood
            for context, next_counts in contexts.items():
                total = sum(next_counts.values())
                for count in next_counts.values():
                    if count > 0:
                        log_L += count * np.log(count / total)
            
            # Calculate number of parameters using observed contexts
            # For each observed context, we have (alphabet_size - 1) free parameters
            # This is more appropriate than theoretical maximum for sparse data
            k = num_observed_contexts * (alphabet_size - 1)

        log_likelihoods.append(log_L)

        if criterion == 'BIC':
            IC = -2 * log_L + k * np.log(n)
        elif criterion == 'AIC':
            IC = -2 * log_L + 2 * k
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        scores.append(IC)

    optimal_order = int(np.argmin(scores))

    transition_matrix = _build_transition_matrix(sequence, optimal_order, alphabet)

    return {
        'optimal_order': optimal_order,
        'scores': scores,
        'log_likelihoods': log_likelihoods,
        'transition_matrix': transition_matrix,
    }


def _build_transition_matrix(sequence, order, alphabet):
    """Build transition probability matrix for given order."""
    if order == 0:
        counts = Counter(sequence)
        total = sum(counts.values())
        probs = {s: counts[s] / total for s in alphabet}
        return probs

    contexts = defaultdict(lambda: defaultdict(int))
    for i in range(order, len(sequence)):
        context = tuple(sequence[i - order:i])
        next_symbol = sequence[i]
        contexts[context][next_symbol] += 1

    transition_probs = {}
    for context, next_counts in contexts.items():
        total = sum(next_counts.values())
        transition_probs[context] = {s: next_counts[s] / total for s in next_counts}

    return transition_probs


def vlmc_fit(
    sequence: list[str],
    max_depth: int = 10,
    pruning_threshold: float = 0.01,
    min_count: int = 2
) -> dict:
    """
    Fit Variable-Length Markov Chain (VLMC) model.

    Parameters
    ----------
    sequence : list of str
        Symbolic sequence.
    max_depth : int, default=10
        Maximum context depth.
    pruning_threshold : float, default=0.01
        Minimum probability threshold for keeping contexts.
    min_count : int, default=2
        Minimum count for context to be considered.

    Returns
    -------
    dict
        Results with keys:
        - 'context_tree': Dict mapping contexts to transition probabilities
        - 'num_contexts': Number of contexts in pruned tree
        - 'max_depth_used': Maximum depth actually used
        - 'alphabet': Set of symbols

    Notes
    -----
    VLMC extends fixed-order Markov chains by allowing variable-length
    contexts based on statistical significance.

    The context tree is built by:
    1. Collecting all contexts up to max_depth
    2. Computing transition probabilities for each context
    3. Pruning contexts with low maximum transition probability

    References
    ----------
    Rissanen, J. (1983). A universal data compression system.
    IEEE Transactions on Information Theory, 29(5), 656-664.

    Bühlmann, P. & Wyner, A.J. (1999). Variable length Markov chains.
    Annals of Statistics, 27(2), 480-513.

    Examples
    --------
    >>> seq = ['A', 'B', 'A', 'C'] * 50
    >>> result = vlmc_fit(seq, max_depth=3)
    >>> result['num_contexts']
    """
    if len(sequence) < max_depth + 1:
        warnings.warn(
            f"Sequence length ({len(sequence)}) should be >> max_depth ({max_depth})",
            UserWarning
        )

    alphabet = set(sequence)
    contexts = defaultdict(lambda: defaultdict(int))

    for i in range(len(sequence)):
        for depth in range(1, min(max_depth + 1, i + 1)):
            context = tuple(sequence[i - depth:i])
            if i < len(sequence):
                next_symbol = sequence[i]
                contexts[context][next_symbol] += 1

    context_tree = {}
    max_depth_used = 0

    for context, next_counts in contexts.items():
        total = sum(next_counts.values())
        if total >= min_count:
            probs = {s: count / total for s, count in next_counts.items()}
            max_prob = max(probs.values())

            if max_prob >= pruning_threshold:
                context_tree[context] = probs
                max_depth_used = max(max_depth_used, len(context))

    return {
        'context_tree': context_tree,
        'num_contexts': len(context_tree),
        'max_depth_used': max_depth_used,
        'alphabet': alphabet,
    }

