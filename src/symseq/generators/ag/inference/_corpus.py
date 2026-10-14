# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Corpus validation for regular-grammar inference.

Generic string-set validation and context counting live in
:mod:`symseq.metrics.grammar._string_sets`; this module adds the restrictions that only
grammar compilation needs.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from symseq.metrics.grammar._string_sets import Context, ContextCounts, validate_string_set

__all__ = ["Context", "ContextCounts", "normalize_corpus"]


def normalize_corpus(
    sequences: Iterable[Sequence[str]],
    *,
    eos: str,
) -> tuple[list[tuple[str, ...]], tuple[str, ...]]:
    """Materialize and validate a corpus without joining sequence boundaries.

    Adds to :func:`~symseq.metrics.grammar._string_sets.validate_string_set` the requirement of an
    EOS marker and the rejection of ArtificialGrammar's reserved indexed-state syntax.
    """
    if not isinstance(eos, str) or not eos:
        raise ValueError("eos must be a non-empty string.")

    corpus, alphabet = validate_string_set(sequences, eos=eos)
    for symbol in alphabet:
        if re.fullmatch(r".*\(\d+\)", symbol):
            raise ValueError(f"Symbol {symbol!r} uses ArtificialGrammar's reserved indexed-state syntax.")
    return corpus, alphabet
