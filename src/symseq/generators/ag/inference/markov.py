# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Fixed-order Markov estimation compiled into regular artificial grammars."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np

from symseq.generators.ag.artificial_grammar import ArtificialGrammar

from ._compiler import compile_context_grammar
from ._corpus import Context, ContextCounts, normalize_corpus


def _fixed_order_counts(
    corpus: Sequence[tuple[str, ...]],
    *,
    order: int,
    eos: str,
) -> tuple[ContextCounts, Counter[str]]:
    counts: defaultdict[Context, Counter[str]] = defaultdict(Counter)
    starts: Counter[str] = Counter()

    for sequence in corpus:
        starts[sequence[0]] += 1
        for index in range(len(sequence)):
            next_symbol = sequence[index + 1] if index + 1 < len(sequence) else eos
            counts[()][next_symbol] += 1
            if order > 0:
                depth = min(order, index + 1)
                counts[sequence[index - depth + 1 : index + 1]][next_symbol] += 1

    return dict(counts), starts


def _bic_scores(
    corpus: Sequence[tuple[str, ...]],
    *,
    max_order: int,
    alphabet_size: int,
    eos: str,
) -> dict[int, float]:
    n_observations = sum(len(sequence) for sequence in corpus)
    outcome_parameters = alphabet_size  # |alphabet + EOS| - 1
    scores: dict[int, float] = {}

    for order in range(max_order + 1):
        counts, _ = _fixed_order_counts(corpus, order=order, eos=eos)
        fitted: ContextCounts
        if order == 0:
            fitted = {(): counts[()]}
        else:
            fitted = {context: next_counts for context, next_counts in counts.items() if context}
        log_likelihood = 0.0
        for next_counts in fitted.values():
            total = sum(next_counts.values())
            for count in next_counts.values():
                log_likelihood += count * np.log(count / total)

        n_parameters = len(fitted) * outcome_parameters
        scores[order] = -2.0 * log_likelihood + n_parameters * np.log(n_observations)

    return scores


def infer_markov(
    sequences: Iterable[Sequence[str]],
    *,
    order: int | Literal["bic", "auto"] = 1,
    max_order: int = 5,
    label: str = "Inferred Markov grammar",
    eos: str = "#",
    rng: np.random.Generator | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> ArtificialGrammar:
    """Estimate a fixed-order Markov model and compile it into an ArtificialGrammar.

    ``order="bic"`` (or the alias ``"auto"``) selects orders from zero through
    ``max_order`` using the Bayesian information criterion. Sequence boundaries
    are preserved and modeled as explicit EOS outcomes.
    """
    corpus, alphabet = normalize_corpus(sequences, eos=eos)

    bic_scores: dict[int, float] | None = None
    if order in {"bic", "auto"}:
        if not isinstance(max_order, int) or isinstance(max_order, bool) or max_order < 0:
            raise ValueError("max_order must be a non-negative integer.")
        bic_scores = _bic_scores(
            corpus,
            max_order=max_order,
            alphabet_size=len(alphabet),
            eos=eos,
        )
        selected_order = min(bic_scores, key=bic_scores.__getitem__)
    elif isinstance(order, int) and not isinstance(order, bool) and order >= 0:
        selected_order = order
    else:
        raise ValueError("order must be a non-negative integer, 'bic', or 'auto'.")

    counts, starts = _fixed_order_counts(corpus, order=selected_order, eos=eos)
    model_contexts: set[Context]
    if selected_order == 0:
        model_contexts = {()}
    else:
        model_contexts = {()} | {context for context in counts if context}

    metadata: dict[str, object] = {
        "method": "markov",
        "order": selected_order,
        "n_sequences": len(corpus),
        "n_observations": sum(len(sequence) for sequence in corpus),
    }
    if bic_scores is not None:
        metadata["criterion"] = "BIC"
        metadata["bic_scores"] = bic_scores

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
