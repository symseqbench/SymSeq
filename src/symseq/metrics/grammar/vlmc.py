# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Variable-length Markov chain (Context algorithm) estimation for string sets."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import chi2  # type: ignore[import-untyped]

from symseq.utils.validation import is_integer, is_real

from ._string_sets import (
    Context,
    boundary_policy,
    iter_targets,
    n_scored_targets,
    suffix_counts,
    validate_string_set,
)


def _context_discrepancy(
    child_counts: Mapping[str, int],
    parent_counts: Mapping[str, int],
) -> float:
    """Return N(child) * KL(p(.|child) || p(.|parent))."""
    child_total = sum(child_counts.values())
    parent_total = sum(parent_counts.values())
    discrepancy = 0.0
    for symbol, count in child_counts.items():
        if count <= 0:
            continue
        child_probability = count / child_total
        parent_probability = parent_counts[symbol] / parent_total
        discrepancy += count * np.log(child_probability / parent_probability)
    return float(discrepancy)


def _terminal_contexts(contexts: set[Context]) -> set[Context]:
    """Non-root contexts with no retained one-symbol-longer extension. Excludes the root."""
    contexts_with_children = {context[1:] for context in contexts if context}
    return {context for context in contexts if context and context not in contexts_with_children}


def _prune_context_tree(
    counts: Mapping[Context, Counter[str]],
    *,
    min_count: int,
    cutoff: float,
) -> tuple[set[Context], dict[Context, float], set[Context]]:
    """Prune terminal contexts bottom-up while their discrepancy from the suffix is below `cutoff`."""
    tree = {context for context, next_counts in counts.items() if sum(next_counts.values()) >= min_count}
    tree.add(())

    discrepancies: dict[Context, float] = {}
    retained_terminals: set[Context] = set()
    pruned: set[Context] = set()

    while True:
        undecided = _terminal_contexts(tree) - retained_terminals
        if not undecided:
            break

        changed = False
        for context in sorted(undecided, key=lambda item: (-len(item), item)):
            parent = context[1:]
            discrepancy = _context_discrepancy(counts[context], counts[parent])
            discrepancies[context] = discrepancy
            if discrepancy < cutoff:
                tree.remove(context)
                pruned.add(context)
                changed = True
            else:
                retained_terminals.add(context)

        if not changed and _terminal_contexts(tree).issubset(retained_terminals):
            break

    return tree, discrepancies, pruned


def _longest_suffix(context: Context, candidates: set[Context]) -> Context:
    """Return the longest suffix of `context` in `candidates`, falling back to the root."""
    for depth in range(len(context), -1, -1):
        suffix = context[-depth:] if depth else ()
        if suffix in candidates:
            return suffix
    raise RuntimeError("The empty fallback context is missing.")


def _resolve_cutoff(
    cutoff: float | str,
    alpha: float | None,
    *,
    n_outcomes: int,
    n_observations: int,
) -> tuple[float, str]:
    """Return the effective pruning cutoff (in nats) and the rule that produced it."""
    if is_real(cutoff):
        return float(cutoff), "explicit"
    rule = "chi2" if alpha is not None else "bic"
    degrees_of_freedom = n_outcomes - 1
    if degrees_of_freedom == 0:
        # A single possible outcome: no extension can change a prediction.
        return math.inf, rule
    if alpha is not None:
        return float(chi2.ppf(1.0 - alpha, df=degrees_of_freedom) / 2.0), rule
    return float(degrees_of_freedom * np.log(n_observations) / 2.0), rule


def _validate_vlmc_settings(max_depth, min_count, cutoff, alpha) -> None:
    if not is_integer(max_depth) or max_depth < 1:
        raise ValueError("max_depth must be a positive integer.")
    if not is_integer(min_count) or min_count < 2:
        raise ValueError("min_count must be an integer of at least 2.")
    if alpha is not None and not (is_real(alpha) and 0.0 < alpha < 1.0):
        raise ValueError("alpha must lie strictly between 0 and 1.")
    if is_real(cutoff):
        if not np.isfinite(cutoff) or cutoff < 0.0:
            raise ValueError("cutoff must be non-negative.")
        if alpha is not None:
            raise ValueError("Pass either alpha (chi-square rule) or a numeric cutoff, not both.")
    elif cutoff != "bic":
        raise ValueError(f"cutoff must be 'bic' or a non-negative number, got {cutoff!r}.")


def vlmc_fit(
    string_set: Iterable[Sequence[str]],
    max_depth: int = 10,
    min_count: int = 2,
    cutoff: float | str = "bic",
    alpha: float | None = None,
    eos: str | None = None,
) -> dict:
    """
    Fit a variable-length Markov chain (VLMC) to a string set with the Context algorithm.

    A suffix tree of all within-string contexts up to `max_depth` is grown and then pruned
    bottom-up: a terminal context is removed when the extra (oldest) symbol does not change its
    next-outcome distribution enough relative to its one-symbol-shorter suffix.

    Parameters
    ----------
    string_set : iterable of sequence of str
        Strings, each a sequence of atomic symbols. A flat list of symbols (a single stream) is
        rejected; see Notes.
    max_depth : int, default=10
        Longest context grown before pruning.
    min_count : int, default=2
        Contexts observed fewer times are not grown. Must be at least 2.
    cutoff : float or 'bic', default='bic'
        Pruning cutoff in nats. ``'bic'`` uses ``(M - 1) ln(n) / 2`` (see Notes). A number is used
        as given; ``0`` disables pruning except for exact suffix duplicates.
    alpha : float or None, default=None
        If given, selects the chi-square rule ``chi2.ppf(1 - alpha, df=M - 1) / 2`` instead of
        ``'bic'``. Cannot be combined with a numeric `cutoff`.
    eos : str or None, default=None
        End-of-string marker. When given, each string's end is an explicit `eos` outcome (the
        complete-string model used by grammar inference). When ``None``, only symbols within
        strings are predicted.

    Returns
    -------
    dict
        ``context_tree`` : dict[tuple of str, dict[str, float]]
            Next-outcome probabilities for every retained context, including the root ``()``.
            Contexts are chronological: the most recent symbol is last.
        ``contexts`` : tuple of tuple of str
            Retained tree nodes, sorted by length.
        ``terminal_contexts`` : tuple of tuple of str
            Retained nodes without retained extensions; ``((),)`` for a root-only tree.
        ``predictive_contexts`` : tuple of tuple of str
            Nodes that are the longest retained suffix of at least one observed target's history.
            Internal nodes can be predictive in an incomplete tree, so this can differ from
            ``terminal_contexts``.
        ``num_nodes``, ``num_terminal_contexts``, ``num_predictive_contexts`` : int
        ``max_depth_used`` : int
            Length of the longest retained context.
        ``cutoff`` : float
            Effective cutoff in nats (``inf`` when there is only one possible outcome).
        ``cutoff_rule`` : {'bic', 'chi2', 'explicit'}
        ``alpha`` : float or None
        ``discrepancies`` : dict[tuple of str, float]
            ``D(w)`` for every context evaluated during pruning.
        ``pruned_contexts`` : tuple of tuple of str
        ``counts`` : dict[tuple of str, Counter]
            Next-outcome counts of every grown context.
        ``start_counts`` : Counter
            First-symbol counts. First symbols are never targets.
        ``alphabet`` : tuple of str
        ``eos`` : str or None
        ``max_depth``, ``min_count``, ``n_strings``, ``n_observations`` : int
        ``boundary_policy`` : {'within_string', 'within_string_eos'}

    Raises
    ------
    TypeError
        If `string_set` is a single stream or contains bare ``str`` elements.
    ValueError
        For invalid settings, `alpha` combined with a numeric `cutoff`, invalid strings, or a
        string set with no scored targets.

    Notes
    -----
    **Boundaries.** No context spans two strings, and strings are never concatenated. The
    targets are the symbols at positions 2..N of each string, plus one `eos` outcome per string
    when `eos` is given. First symbols are not targets. Contexts near a string start are
    truncated to the available history. Single streams are not supported yet; ``[stream]``
    treats a stream as one string.

    **Pruning statistic.** For a terminal context w with suffix s = w[1:] (oldest symbol removed),

    .. math::

        D(w) = N(w) \\, \\mathrm{KL}\\big(\\hat p(\\cdot \\mid w) \\,\\|\\, \\hat p(\\cdot \\mid s)\\big)
             = \\sum_a N(w, a) \\ln \\frac{\\hat p(a \\mid w)}{\\hat p(a \\mid s)}.

    w is pruned when D(w) < `cutoff`, repeatedly from the leaves upward. D(w) is the *local*
    log-likelihood gain of predicting w's continuations with w instead of s; it is not a
    likelihood-ratio statistic for refitting the whole tree.

    **Cutoff.** M is the number of outcomes (symbols, plus `eos` when given) and n the number of
    targets. The default ``'bic'`` cutoff ``(M - 1) ln(n) / 2`` is half the BIC penalty for the
    M - 1 extra parameters of a split, in the same units as D(w). It grows with n, so the number of
    spurious contexts does not grow with the number of comparisons as it does under a fixed-alpha
    chi-square rule. It is a heuristic *local* rule, not the global BIC context-tree estimator of
    Csiszár & Talata (2006), and no cutoff guarantees recovery of the true tree (for example,
    the root alone for an independent source) on every finite sample. Sparse counts make D(w)
    noisy.

    **Measured behavior** (100 seeded string sets per setting, strings of length 5-10,
    ``max_depth=4``; see ``docs/INTERNAL/refactoring/markov_vlmc_explained.md`` §13.3):

    - On i.i.d. input with 100-500 strings, the BIC-type cutoff recovered the root-only tree in
      98-100% of runs. The chi-square rule with ``alpha=0.05`` did so in 11-13% and kept about 8
      spurious nodes per run.
    - The BIC-type cutoff **underfits weak signals** at small n. On an order-1 chain whose rows
      differ only modestly from the marginal, it pruned about one true context per run at
      20-100 strings, and recovered the full tree in 86% of runs only at 500 strings.
    - Neither rule reliably recovered order-2 or variable-length trees with 20 strings.

    A pruned context therefore means "not supported at this sample size", not "absent".
    Compare ``discrepancies`` with ``cutoff`` for borderline contexts.

    References
    ----------
    Rissanen, J. (1983). A universal data compression system. IEEE Transactions on Information
    Theory, 29(5), 656-664.

    Bühlmann, P., & Wyner, A. J. (1999). Variable length Markov chains. Annals of Statistics,
    27(2), 480-513.

    Csiszár, I., & Talata, Zs. (2006). Context tree estimation for not necessarily finite memory
    processes, via BIC and MDL. IEEE Transactions on Information Theory, 52(3), 1007-1016.

    Examples
    --------
    >>> strings = [['A', 'B', 'X']] * 50 + [['C', 'B', 'Y']] * 50
    >>> fit = vlmc_fit(strings, max_depth=2)
    >>> sorted(fit['terminal_contexts'])
    [('A',), ('A', 'B'), ('C',), ('C', 'B')]
    """
    _validate_vlmc_settings(max_depth, min_count, cutoff, alpha)
    corpus, alphabet = validate_string_set(string_set, eos=eos)
    n_observations = n_scored_targets(corpus, eos)
    if n_observations == 0:
        raise ValueError("Insufficient data: the string set has no scored targets (all strings have length 1).")

    max_depth = int(max_depth)
    counts, starts = suffix_counts(corpus, max_depth=max_depth, eos=eos)
    effective_cutoff, cutoff_rule = _resolve_cutoff(
        cutoff,
        alpha,
        n_outcomes=len(alphabet) + (eos is not None),
        n_observations=n_observations,
    )
    model_contexts, discrepancies, pruned = _prune_context_tree(
        counts,
        min_count=int(min_count),
        cutoff=effective_cutoff,
    )

    def by_length(item: Context) -> tuple[int, Context]:
        return (len(item), item)

    terminals = _terminal_contexts(model_contexts) or {()}
    predictive = {
        _longest_suffix(sequence[: index + 1][-max_depth:], model_contexts)
        for sequence in corpus
        for index, _ in iter_targets(sequence, eos)
    }
    context_tree = {
        context: {outcome: count / sum(counts[context].values()) for outcome, count in counts[context].items()}
        for context in sorted(model_contexts, key=by_length)
    }

    return {
        "context_tree": context_tree,
        "contexts": tuple(sorted(model_contexts, key=by_length)),
        "terminal_contexts": tuple(sorted(terminals, key=by_length)),
        "predictive_contexts": tuple(sorted(predictive, key=by_length)),
        "num_nodes": len(model_contexts),
        "num_terminal_contexts": len(terminals),
        "num_predictive_contexts": len(predictive),
        "max_depth_used": max(len(context) for context in model_contexts),
        "cutoff": effective_cutoff,
        "cutoff_rule": cutoff_rule,
        "alpha": alpha,
        "discrepancies": discrepancies,
        "pruned_contexts": tuple(sorted(pruned, key=by_length)),
        "counts": counts,
        "start_counts": starts,
        "alphabet": alphabet,
        "eos": eos,
        "max_depth": max_depth,
        "min_count": int(min_count),
        "n_strings": len(corpus),
        "n_observations": n_observations,
        "boundary_policy": boundary_policy(eos),
    }
