# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Corpus validation and context counting for regular-grammar inference."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

Context = tuple[str, ...]
ContextCounts = dict[Context, Counter[str]]


def normalize_corpus(
    sequences: Iterable[Sequence[str]],
    *,
    eos: str,
) -> tuple[list[tuple[str, ...]], tuple[str, ...]]:
    """Materialize and validate a corpus without joining sequence boundaries."""
    if not isinstance(eos, str) or not eos:
        raise ValueError("eos must be a non-empty string.")

    corpus: list[tuple[str, ...]] = []
    alphabet: dict[str, None] = {}
    for sequence_index, sequence in enumerate(sequences):
        materialized = tuple(sequence)
        if not materialized:
            raise ValueError(f"Sequence {sequence_index} is empty; ArtificialGrammar cannot model empty strings.")
        for symbol in materialized:
            if not isinstance(symbol, str) or not symbol:
                raise ValueError("Every corpus symbol must be a non-empty string.")
            if symbol == eos:
                raise ValueError(
                    f"Corpus sequences must not contain the EOS marker {eos!r}; boundaries are added internally."
                )
            if re.fullmatch(r".*\(\d+\)", symbol):
                raise ValueError(f"Symbol {symbol!r} uses ArtificialGrammar's reserved indexed-state syntax.")
            alphabet.setdefault(symbol, None)
        corpus.append(materialized)

    if not corpus:
        raise ValueError("Cannot infer a grammar from an empty corpus.")
    return corpus, tuple(alphabet)


def collect_context_counts(
    corpus: Sequence[tuple[str, ...]],
    *,
    max_depth: int,
    eos: str,
) -> tuple[ContextCounts, Counter[str]]:
    """Count all observed suffix contexts, starts, and explicit EOS outcomes."""
    counts: defaultdict[Context, Counter[str]] = defaultdict(Counter)
    starts: Counter[str] = Counter()

    for sequence in corpus:
        starts[sequence[0]] += 1
        for index in range(len(sequence)):
            next_symbol = sequence[index + 1] if index + 1 < len(sequence) else eos
            counts[()][next_symbol] += 1
            history = sequence[: index + 1]
            for depth in range(1, min(max_depth, len(history)) + 1):
                counts[history[-depth:]][next_symbol] += 1

    return dict(counts), starts
