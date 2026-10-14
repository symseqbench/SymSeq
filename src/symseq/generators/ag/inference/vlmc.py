# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Variable-length Markov estimation using the classical Context algorithm."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from symseq.generators.ag.artificial_grammar import ArtificialGrammar
from symseq.metrics.grammar.vlmc import vlmc_fit

from ._compiler import compile_context_grammar
from ._corpus import normalize_corpus


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

    Growth and pruning are delegated to :func:`symseq.metrics.grammar.vlmc_fit`
    (with ``eos=eos``): a maximal observed suffix tree is grown, and its terminal
    contexts are compared with their one-symbol-shorter suffixes using
    ``N(w) * KL(p(.|w) || p(.|suffix(w)))`` and pruned bottom-up.

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
    if cutoff is not None:
        fit = vlmc_fit(corpus, max_depth=max_depth, min_count=min_count, cutoff=cutoff, eos=eos)
    else:
        fit = vlmc_fit(corpus, max_depth=max_depth, min_count=min_count, alpha=alpha, eos=eos)

    metadata = {
        "method": "vlmc_context",
        "max_depth": max_depth,
        "min_count": min_count,
        "alpha": alpha,
        "cutoff": fit["cutoff"],
        "n_sequences": fit["n_strings"],
        "n_observations": fit["n_observations"],
        "discrepancies": fit["discrepancies"],
        "pruned_contexts": fit["pruned_contexts"],
    }
    return compile_context_grammar(
        counts=fit["counts"],
        model_contexts=set(fit["contexts"]),
        alphabet=alphabet,
        start_counts=fit["start_counts"],
        eos=eos,
        label=label,
        inference_metadata=metadata,
        rng=rng,
        seed=seed,
        verbose=verbose,
    )
