# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Boundary-aware Markov context counts and order selection for string sets."""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from symseq.utils.validation import is_integer

from ._string_sets import (
    Context,
    boundary_policy,
    fixed_order_counts,
    n_scored_targets,
    suffix_counts,
    validate_string_set,
)


def context_counts(string_set: Iterable[Sequence[str]], *, max_depth: int, eos: str | None = None) -> dict:
    """
    Count next-symbol outcomes for every within-string context up to a maximum depth.

    Parameters
    ----------
    string_set : iterable of sequence of str
        Strings, each a sequence of atomic symbols. A flat list of symbols (a single stream) is
        rejected; see Notes.
    max_depth : int
        Longest context counted. ``0`` counts only the root ``()``.
    eos : str or None, default=None
        End-of-string marker. When given, the end of every string is counted as an explicit
        `eos` outcome. When ``None``, only symbols within strings are targets.

    Returns
    -------
    dict
        ``counts`` : dict[tuple of str, Counter]
            Next-outcome counts for each observed context, including the root ``()``. Contexts
            are chronological: the most recent symbol is last.
        ``start_counts`` : Counter
            First-symbol counts. First symbols are never targets.
        ``alphabet`` : tuple of str
            Distinct symbols in order of first occurrence, without `eos`.
        ``n_strings`` : int
        ``n_observations`` : int
            Number of scored targets, ``sum(len(s) - 1)`` plus one per string when `eos` is set.
        ``boundary_policy`` : {'within_string', 'within_string_eos'}

    Notes
    -----
    **No context spans two strings.** At position i of a string, the counted contexts are the
    suffixes of that string's first i+1 symbols, so contexts near a string start are shorter.
    Strings are never concatenated. Single streams are not supported yet; ``[stream]`` treats a
    stream as one string.
    """
    if not is_integer(max_depth) or max_depth < 0:
        raise ValueError("max_depth must be a non-negative integer.")
    corpus, alphabet = validate_string_set(string_set, eos=eos)
    counts, starts = suffix_counts(corpus, max_depth=int(max_depth), eos=eos)
    return {
        "counts": counts,
        "start_counts": starts,
        "alphabet": alphabet,
        "n_strings": len(corpus),
        "n_observations": n_scored_targets(corpus, eos),
        "boundary_policy": boundary_policy(eos),
    }


def _log_likelihood(fitted: Mapping[Context, Counter]) -> float:
    """Maximized log-likelihood ``sum N(w,a) ln(N(w,a) / N(w))`` in nats."""
    log_likelihood = 0.0
    for next_counts in fitted.values():
        total = sum(next_counts.values())
        for count in next_counts.values():
            log_likelihood += count * np.log(count / total)
    return log_likelihood


def _fitted_contexts(counts: Mapping[Context, Counter], order: int) -> dict[Context, Counter]:
    """Contexts whose rows form the order-`order` model: the root alone at order 0."""
    if order == 0:
        return {(): counts[()]}
    return {context: next_counts for context, next_counts in counts.items() if context}


def markov_order_selection(
    string_set: Iterable[Sequence[str]],
    max_order: int = 5,
    criterion: str = "BIC",
    eos: str | None = None,
) -> dict:
    """
    Select a Markov order for a string set with an information criterion.

    Every candidate order is fitted by maximum likelihood and scored on the **same** targets,
    so the likelihoods are comparable across orders.

    Parameters
    ----------
    string_set : iterable of sequence of str
        Strings, each a sequence of atomic symbols. A flat list of symbols (a single stream) is
        rejected; see Notes.
    max_order : int, default=5
        Largest candidate order. Orders ``0..max_order`` are scored. It is never lowered
        silently: if it exceeds the longest usable context, the extra orders are still scored
        (they reproduce the fit of the effective maximum and lose ties) and a warning is issued.
    criterion : {'BIC', 'AIC'}, default='BIC'
        Information criterion. Lower scores are better.
    eos : str or None, default=None
        End-of-string marker. When given, each string's end is scored as an explicit `eos`
        outcome (the complete-string generation model used by grammar inference). When
        ``None``, only symbols within strings are predicted.

    Returns
    -------
    dict
        ``optimal_order`` : int
            Order with the lowest score; ties go to the lower order.
        ``scores``, ``log_likelihoods``, ``n_parameters`` : list
            Per-order values, indexed by order. Log-likelihoods are in nats.
        ``n_observations`` : int
            Number of scored targets, identical for every order.
        ``n_strings`` : int
        ``criterion`` : {'BIC', 'AIC'}
        ``selected_at_cap`` : bool
            Whether the selected order equals `max_order`. If so, longer memory was not ruled out.
        ``parameter_count_policy`` : {'observed_contexts'}
        ``boundary_policy`` : {'within_string', 'within_string_eos'}
        ``transition_probabilities`` : dict[tuple of str, dict[str, float]]
            Next-outcome probabilities of the selected model. These are the rows that produced
            its log-likelihood, not a refit.
        ``start_counts`` : Counter
            First-symbol counts, which are not part of any score.

    Raises
    ------
    TypeError
        If `string_set` is a single stream or contains bare ``str`` elements.
    ValueError
        For invalid `max_order` or `criterion`, invalid strings, or a string set with no
        scored targets (for example, only length-1 strings with ``eos=None``).

    Notes
    -----
    **Boundaries.** No context spans two strings, and strings are never concatenated. The
    targets are the symbols at positions 2..N of each string, plus one `eos` outcome per string
    when `eos` is given. First symbols are not scored; their distribution is the same for every
    candidate and does not affect the comparison. At a target with fewer than k preceding
    symbols in its string, the order-k candidate uses the available history. The order-k
    candidate is therefore a context tree complete to depth k, with shorter contexts at string
    starts. Because no target is dropped, every candidate scores all ``n`` targets. Single
    streams are not supported yet; ``[stream]`` treats a stream as one string (with ``eos``, it
    adds one end outcome and its parameters).

    **Scores.** With counts N(w, a) of outcome a after context w,

    .. math::

        \\ell_k = \\sum_{w, a} N(w, a) \\ln \\frac{N(w, a)}{N(w)}, \\qquad
        \\mathrm{BIC}_k = -2 \\ell_k + d_k \\ln n, \\qquad
        \\mathrm{AIC}_k = -2 \\ell_k + 2 d_k,

    with ``d_k = C_k (M - 1)``, where C_k is the number of contexts *observed* at order k and M
    the number of outcomes (symbols, plus `eos` when given). This observed-context count is a
    data-dependent variant of the unrestricted dimension ``|A|^k (|A| - 1)``; the standard BIC
    consistency result (Csiszár & Shields, 2000) does not transfer to it automatically.

    **Reliability.** On short string sets, selection is unreliable in both directions:
    overfitting from sampling noise and underfitting because the penalty dominates. Inspect
    ``scores`` rather than relying on ``optimal_order`` alone.

    References
    ----------
    Schwarz, G. (1978). Estimating the dimension of a model. Annals of Statistics, 6(2), 461-464.

    Csiszár, I., & Shields, P. C. (2000). The consistency of the BIC Markov order estimator.
    Annals of Statistics, 28(6), 1601-1619.

    Examples
    --------
    >>> strings = [['A', 'B'] * 10 for _ in range(20)]
    >>> markov_order_selection(strings, max_order=3)['optimal_order']
    1
    """
    if not is_integer(max_order) or max_order < 0:
        raise ValueError("max_order must be a non-negative integer.")
    if criterion not in {"BIC", "AIC"}:
        raise ValueError(f"criterion must be 'BIC' or 'AIC', got {criterion!r}.")
    max_order = int(max_order)

    corpus, alphabet = validate_string_set(string_set, eos=eos)
    n_observations = n_scored_targets(corpus, eos)
    if n_observations == 0:
        raise ValueError("Insufficient data: the string set has no scored targets (all strings have length 1).")

    # Longest context any target can use: its preceding symbols in the same string.
    effective_max_order = max(len(sequence) for sequence in corpus) - (1 if eos is None else 0)
    if max_order > effective_max_order:
        warnings.warn(
            f"max_order={max_order} exceeds the longest usable context ({effective_max_order}); "
            "higher orders reproduce that fit and lose ties.",
            UserWarning,
            stacklevel=2,
        )

    outcome_parameters = len(alphabet) + (eos is not None) - 1
    scores: list[float] = []
    log_likelihoods: list[float] = []
    n_parameters: list[int] = []
    fits: list[dict[Context, Counter]] = []
    for order in range(max_order + 1):
        counts, _ = fixed_order_counts(corpus, order=order, eos=eos)
        fitted = _fitted_contexts(counts, order)
        log_likelihood = _log_likelihood(fitted)
        parameters = len(fitted) * outcome_parameters
        if criterion == "BIC":
            score = -2.0 * log_likelihood + parameters * np.log(n_observations)
        else:
            score = -2.0 * log_likelihood + 2.0 * parameters
        scores.append(float(score))
        log_likelihoods.append(float(log_likelihood))
        n_parameters.append(parameters)
        fits.append(fitted)

    optimal_order = min(range(max_order + 1), key=scores.__getitem__)
    transition_probabilities = {
        context: {outcome: count / sum(next_counts.values()) for outcome, count in next_counts.items()}
        for context, next_counts in fits[optimal_order].items()
    }
    _, starts = fixed_order_counts(corpus, order=0, eos=eos)

    return {
        "optimal_order": optimal_order,
        "scores": scores,
        "log_likelihoods": log_likelihoods,
        "n_parameters": n_parameters,
        "n_observations": n_observations,
        "n_strings": len(corpus),
        "criterion": criterion,
        "selected_at_cap": optimal_order == max_order,
        "parameter_count_policy": "observed_contexts",
        "boundary_policy": boundary_policy(eos),
        "transition_probabilities": transition_probabilities,
        "start_counts": starts,
    }
