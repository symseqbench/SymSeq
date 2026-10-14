# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Boundary-aware string-set validation and context counting for grammar-level metrics.

Every function here treats a string set as separate strings. No context ever spans two
strings: a context is always a suffix of the history *within* the current string, and is
truncated at the string start. Grammar inference (``generators/ag/inference``) builds on the
same helpers.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

Context = tuple[str, ...]
ContextCounts = dict[Context, Counter[str]]


def validate_string_set(
    string_set: Iterable[Sequence[str]],
    *,
    eos: str | None,
) -> tuple[list[tuple[str, ...]], tuple[str, ...]]:
    """
    Materialize and validate a string set without joining string boundaries.

    Parameters
    ----------
    string_set : iterable of sequence of str
        Strings, each a sequence of atomic, non-empty ``str`` symbols.
    eos : str or None
        End-of-string marker, or ``None`` when no end-of-string outcome is modeled. When given,
        it must not occur inside any string.

    Returns
    -------
    corpus : list of tuple of str
        The strings, materialized as tuples.
    alphabet : tuple of str
        Distinct symbols in order of first occurrence. ``eos`` is not included.

    Raises
    ------
    TypeError
        If `string_set` is a ``str`` or contains a bare ``str`` element. A flat list of symbols
        is a single stream, which is not supported; a bare ``str`` element would otherwise be
        split silently into characters.
    ValueError
        If `eos` is invalid, or the string set, a string, or a symbol is empty, a symbol is not a
        ``str``, or `eos` occurs inside a string.
    """
    if eos is not None and (not isinstance(eos, str) or not eos):
        raise ValueError("eos must be None or a non-empty string.")
    if isinstance(string_set, str):
        raise TypeError("string_set must be a collection of strings (sequences of symbols), not a str.")

    corpus: list[tuple[str, ...]] = []
    alphabet: dict[str, None] = {}
    for string_index, string in enumerate(string_set):
        if isinstance(string, str):
            raise TypeError(
                "string_set must contain sequences of symbols, e.g. [['A', 'B'], ['A', 'C']], but element "
                f"{string_index} is a bare str ({string!r}). Single streams (a flat list of symbols) are not "
                "supported; pass separate strings, or [stream] to treat one stream as a single string."
            )
        materialized = tuple(string)
        if not materialized:
            raise ValueError(f"Sequence {string_index} is empty; empty strings cannot be modeled.")
        for symbol in materialized:
            if not isinstance(symbol, str) or not symbol:
                raise ValueError("Every corpus symbol must be a non-empty string.")
            if eos is not None and symbol == eos:
                raise ValueError(
                    f"Corpus sequences must not contain the EOS marker {eos!r}; boundaries are added internally."
                )
            alphabet.setdefault(symbol, None)
        corpus.append(materialized)

    if not corpus:
        raise ValueError("Cannot analyze an empty string set.")
    return corpus, tuple(alphabet)


def iter_targets(sequence: tuple[str, ...], eos: str | None):
    """Yield ``(index, next_outcome)`` for every scored target of one string.

    The target after position ``index`` is the next symbol of the same string, or `eos` after
    the last symbol when `eos` is given. The first symbol of a string is never a target.
    """
    for index in range(len(sequence)):
        if index + 1 < len(sequence):
            yield index, sequence[index + 1]
        elif eos is not None:
            yield index, eos


def n_scored_targets(corpus: Sequence[tuple[str, ...]], eos: str | None) -> int:
    """Return the number of scored targets: ``sum(len(s) - 1)``, plus one per string with `eos`."""
    return sum(len(sequence) - 1 for sequence in corpus) + (len(corpus) if eos is not None else 0)


def suffix_counts(
    corpus: Sequence[tuple[str, ...]],
    *,
    max_depth: int,
    eos: str | None,
) -> tuple[ContextCounts, Counter[str]]:
    """Count every within-string suffix context up to `max_depth`, plus start symbols.

    Each target is counted under the root ``()`` and under every suffix of its history of length
    1 to ``min(max_depth, available history)``.
    """
    counts: defaultdict[Context, Counter[str]] = defaultdict(Counter)
    starts: Counter[str] = Counter()

    for sequence in corpus:
        starts[sequence[0]] += 1
        for index, next_symbol in iter_targets(sequence, eos):
            counts[()][next_symbol] += 1
            history = sequence[: index + 1]
            for depth in range(1, min(max_depth, len(history)) + 1):
                counts[history[-depth:]][next_symbol] += 1

    return dict(counts), starts


def fixed_order_counts(
    corpus: Sequence[tuple[str, ...]],
    *,
    order: int,
    eos: str | None,
) -> tuple[ContextCounts, Counter[str]]:
    """Count order-`order` contexts, truncated at string starts, plus start symbols.

    Each target is counted under the root ``()`` and under the last ``min(order, available
    history)`` symbols of its own string.
    """
    counts: defaultdict[Context, Counter[str]] = defaultdict(Counter)
    starts: Counter[str] = Counter()

    for sequence in corpus:
        starts[sequence[0]] += 1
        for index, next_symbol in iter_targets(sequence, eos):
            counts[()][next_symbol] += 1
            if order > 0:
                depth = min(order, index + 1)
                counts[sequence[index - depth + 1 : index + 1]][next_symbol] += 1

    return dict(counts), starts


def boundary_policy(eos: str | None) -> str:
    """Name the boundary policy reported in metric results."""
    return "within_string" if eos is None else "within_string_eos"
