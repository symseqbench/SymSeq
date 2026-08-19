# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Variable-length Markov estimation using the classical Context algorithm."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import chi2  # type: ignore[import-untyped]

from symseq.generators.ag.artificial_grammar import ArtificialGrammar

from ._compiler import compile_context_grammar
from ._corpus import Context, collect_context_counts, normalize_corpus


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
    contexts_with_children = {context[1:] for context in contexts if context}
    return {context for context in contexts if context and context not in contexts_with_children}


def _prune_context_tree(
    counts: Mapping[Context, Counter[str]],
    *,
    min_count: int,
    cutoff: float,
) -> tuple[set[Context], dict[Context, float], set[Context]]:
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


def infer_vlmc(
    sequences: Iterable[Sequence[str]],
    *,
    max_depth: int = 10,
    min_count: int = 2,
    alpha: float = 0.05,
    cutoff: float | None = None,
    label: str = "Inferred VLMC grammar",
    eos: str = "#",
    rng: np.random.Generator | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> ArtificialGrammar:
    """Estimate a VLMC with Rissanen/Buehlmann-Wyner Context pruning.

    A maximal observed suffix tree is grown first. Its terminal contexts are
    compared with their one-symbol-shorter suffixes using the log-likelihood
    ratio ``N(w) * KL(p(.|w) || p(.|suffix(w)))`` and pruned bottom-up.

    When ``cutoff`` is omitted, the practical chi-square rule used by established
    VLMC implementations is applied: half the ``1 - alpha`` quantile with
    ``|alphabet + EOS| - 1`` degrees of freedom.

    References
    ----------
    Rissanen, J. (1983). A universal data compression system.
    IEEE Transactions on Information Theory, 29(5), 656-664.

    Buehlmann, P., & Wyner, A. J. (1999). Variable length Markov chains.
    The Annals of Statistics, 27(2), 480-513.
    """
    if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 1:
        raise ValueError("max_depth must be a positive integer.")
    if not isinstance(min_count, int) or isinstance(min_count, bool) or min_count < 2:
        raise ValueError("min_count must be an integer of at least 2.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1.")
    if cutoff is not None and cutoff < 0.0:
        raise ValueError("cutoff must be non-negative.")

    corpus, alphabet = normalize_corpus(sequences, eos=eos)
    counts, starts = collect_context_counts(corpus, max_depth=max_depth, eos=eos)

    degrees_of_freedom = len(alphabet)  # |alphabet + EOS| - 1
    effective_cutoff = (
        float(cutoff) if cutoff is not None else float(chi2.ppf(1.0 - alpha, df=degrees_of_freedom) / 2.0)
    )
    model_contexts, discrepancies, pruned = _prune_context_tree(
        counts,
        min_count=min_count,
        cutoff=effective_cutoff,
    )

    metadata = {
        "method": "vlmc_context",
        "max_depth": max_depth,
        "min_count": min_count,
        "alpha": alpha,
        "cutoff": effective_cutoff,
        "n_sequences": len(corpus),
        "n_observations": sum(len(sequence) for sequence in corpus),
        "discrepancies": discrepancies,
        "pruned_contexts": tuple(sorted(pruned, key=lambda item: (len(item), item))),
    }
    return compile_context_grammar(
        counts=counts,
        model_contexts=model_contexts,
        alphabet=alphabet,
        start_counts=starts,
        eos=eos,
        label=label,
        inference_metadata=metadata,
        rng=rng,
        seed=seed,
        verbose=verbose,
    )
