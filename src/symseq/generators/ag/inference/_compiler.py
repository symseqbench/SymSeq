# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Compile probabilistic suffix contexts into an :class:`ArtificialGrammar`."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from symseq.generators.ag.artificial_grammar import ArtificialGrammar

from ._corpus import Context, ContextCounts


def _longest_suffix(context: Context, candidates: set[Context]) -> Context:
    for depth in range(len(context), -1, -1):
        suffix = context[-depth:] if depth else ()
        if suffix in candidates:
            return suffix
    raise RuntimeError("The empty fallback context is missing.")


def _state_names(contexts: set[Context]) -> dict[Context, str]:
    grouped: defaultdict[str, list[Context]] = defaultdict(list)
    for context in contexts:
        grouped[context[-1]].append(context)

    names: dict[Context, str] = {}
    for symbol, symbol_contexts in grouped.items():
        ordered = sorted(symbol_contexts, key=lambda item: (len(item), item))
        if len(ordered) == 1:
            names[ordered[0]] = symbol
        else:
            for index, context in enumerate(ordered):
                names[context] = f"{symbol}({index})"
    return names


def compile_context_grammar(
    *,
    counts: ContextCounts,
    model_contexts: set[Context],
    alphabet: Sequence[str],
    start_counts: Counter[str],
    eos: str,
    label: str,
    inference_metadata: Mapping[str, Any],
    rng: np.random.Generator | None,
    seed: int,
    verbose: bool,
) -> ArtificialGrammar:
    """Compile context distributions and suffix updates into a node-emitting FSA."""
    if () not in model_contexts:
        raise ValueError("model_contexts must contain the empty fallback context.")

    # Tree nodes carry predictive memory. Singleton fallbacks are required because
    # an ArtificialGrammar node also emits the current symbol.
    memory_contexts = {context for context in model_contexts if context}
    memory_contexts.update((symbol,) for symbol in alphabet)
    names = _state_names(memory_contexts)

    transitions: list[tuple[str, str, float]] = []
    terminal_states: list[str] = []
    prediction_contexts: dict[Context, Context] = {}

    for memory in sorted(memory_contexts, key=lambda item: (len(item), item)):
        predictor = _longest_suffix(memory, model_contexts)
        prediction_contexts[memory] = predictor
        next_counts = counts[predictor]
        total = sum(next_counts.values())
        if total <= 0:
            raise RuntimeError(f"Context {predictor!r} has no transition observations.")

        for next_symbol, count in sorted(next_counts.items()):
            probability = count / total
            if next_symbol == eos:
                transitions.append((names[memory], eos, probability))
                terminal_states.append(names[memory])
                continue

            extended = (*memory, next_symbol)
            target_memory = _longest_suffix(extended, memory_contexts)
            transitions.append((names[memory], names[target_memory], probability))

    total_starts = sum(start_counts.values())
    start_probabilities = {names[(symbol,)]: count / total_starts for symbol, count in start_counts.items()}
    start_states = list(start_probabilities)

    metadata = {
        "inference": {
            **dict(inference_metadata),
            "contexts": tuple(sorted(model_contexts, key=lambda item: (len(item), item))),
            "state_contexts": {name: context for context, name in names.items()},
            "prediction_contexts": {names[memory]: predictor for memory, predictor in prediction_contexts.items()},
            "start_probabilities": dict(start_probabilities),
        }
    }
    states = [names[context] for context in sorted(memory_contexts, key=lambda item: (len(item), item))]
    return ArtificialGrammar(
        label=label,
        states=states,
        alphabet=list(alphabet),
        transitions=transitions,
        start_states=start_states,
        start_probabilities=start_probabilities,
        terminal_states=sorted(set(terminal_states)),
        eos=eos,
        rng=rng,
        seed=seed,
        metadata=metadata,
        verbose=verbose,
    )
