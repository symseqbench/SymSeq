# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Fixed-order Markov estimation compiled into regular artificial grammars."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np

from symseq.generators.ag.artificial_grammar import ArtificialGrammar
from symseq.metrics.grammar._string_sets import fixed_order_counts
from symseq.metrics.grammar.markov import markov_order_selection
from symseq.utils.validation import is_integer

from ._compiler import compile_context_grammar
from ._corpus import Context, normalize_corpus


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
    ``max_order`` with :func:`symseq.metrics.grammar.markov_order_selection` (BIC,
    ``eos=eos``). Sequence boundaries are preserved and modeled as explicit EOS outcomes.
    """
    corpus, alphabet = normalize_corpus(sequences, eos=eos)

    bic_scores: dict[int, float] | None = None
    if order in {"bic", "auto"}:
        if not is_integer(max_order) or max_order < 0:
            raise ValueError("max_order must be a non-negative integer.")
        selection = markov_order_selection(corpus, max_order=max_order, criterion="BIC", eos=eos)
        bic_scores = dict(enumerate(selection["scores"]))
        selected_order = selection["optimal_order"]
    elif is_integer(order) and order >= 0:
        selected_order = order
    else:
        raise ValueError("order must be a non-negative integer, 'bic', or 'auto'.")

    counts, starts = fixed_order_counts(corpus, order=selected_order, eos=eos)
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
