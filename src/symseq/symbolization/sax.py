# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Symbolic Aggregate approXimation (SAX) backed by :mod:`saxpy`.

``saxpy`` is an optional GPL-2.0-only dependency. It is imported only when a
SAX transform is executed; SymSeq does not vendor or modify it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np

from symseq.utils.validation import is_integer

_SAXPY_ALPHABET = "abcdefghijklmnopqrstuvwxyz"
_DEFAULT_SYMBOLS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _require_saxpy() -> Callable[..., str]:
    try:
        from saxpy.sax import sax_by_chunking  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "saxpy is required for SAX symbolization. Install SymSeq with: pip install 'symseq[sax]'"
        ) from exc
    return cast("Callable[..., str]", sax_by_chunking)


def _validate_series(series: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(series, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"Expected a one-dimensional series, got shape {values.shape}.")
    if values.size == 0:
        raise ValueError("Cannot symbolize an empty series.")
    if not np.all(np.isfinite(values)):
        raise ValueError("SAX input must contain only finite values.")
    return values


def sax_symbolize(
    series: Sequence[float] | np.ndarray,
    paa_size: int,
    alphabet_size: int = 4,
    *,
    znorm_threshold: float = 0.01,
    symbols: Sequence[str] | None = None,
) -> list[str]:
    """Convert one continuous series to a SAX symbol sequence.

    Parameters
    ----------
    series
        One-dimensional continuous signal.
    paa_size
        Number of Piecewise Aggregate Approximation segments and therefore
        the number of output symbols.
    alphabet_size
        Number of Gaussian SAX bins.
    znorm_threshold
        Standard-deviation threshold used by ``saxpy`` during z-normalization.
    symbols
        Optional output alphabet. By default, SAX's lowercase letters are
        mapped to ``A``, ``B``, ... for consistency with SymSeq grammars.

    Returns
    -------
    list of str
        Symbolic sequence of length ``paa_size``.

    References
    ----------
    Lin, J., Keogh, E., Lonardi, S., & Chiu, B. (2003). A symbolic
    representation of time series, with implications for streaming algorithms.
    """
    if not is_integer(paa_size) or paa_size < 1:
        raise ValueError("paa_size must be a positive integer.")
    if not is_integer(alphabet_size) or alphabet_size < 2:
        raise ValueError("alphabet_size must be an integer of at least 2.")
    if znorm_threshold < 0:
        raise ValueError("znorm_threshold must be non-negative.")

    output_symbols = tuple(symbols) if symbols is not None else _DEFAULT_SYMBOLS[:alphabet_size]
    if len(output_symbols) != alphabet_size:
        raise ValueError(f"Expected exactly {alphabet_size} output symbols, got {len(output_symbols)}.")
    if len(set(output_symbols)) != len(output_symbols):
        raise ValueError("SAX output symbols must be unique.")
    if not all(isinstance(symbol, str) and symbol for symbol in output_symbols):
        raise ValueError("Every SAX output symbol must be a non-empty string.")
    if alphabet_size > len(_SAXPY_ALPHABET):
        raise ValueError(f"alphabet_size cannot exceed {len(_SAXPY_ALPHABET)} with the saxpy backend.")

    values = _validate_series(series)
    sax_by_chunking = _require_saxpy()
    word = sax_by_chunking(
        values,
        paa_size=paa_size,
        alphabet_size=alphabet_size,
        znorm_threshold=znorm_threshold,
    )

    symbol_map = dict(zip(_SAXPY_ALPHABET[:alphabet_size], output_symbols, strict=True))
    try:
        result = [symbol_map[letter] for letter in word]
    except KeyError as exc:
        raise RuntimeError(f"saxpy returned an unexpected symbol: {exc.args[0]!r}.") from exc

    if len(result) != paa_size:
        raise RuntimeError(f"saxpy returned {len(result)} symbols, expected paa_size={paa_size}.")
    return result


@dataclass(frozen=True)
class SAXSymbolizer:
    """Reusable configuration for whole-series SAX symbolization."""

    paa_size: int
    alphabet_size: int = 4
    znorm_threshold: float = 0.01
    symbols: tuple[str, ...] | None = None

    def transform(self, series: Sequence[float] | np.ndarray) -> list[str]:
        """Symbolize one continuous series."""
        return sax_symbolize(
            series,
            paa_size=self.paa_size,
            alphabet_size=self.alphabet_size,
            znorm_threshold=self.znorm_threshold,
            symbols=self.symbols,
        )

    def transform_many(
        self,
        series: Iterable[Sequence[float] | np.ndarray],
    ) -> list[list[str]]:
        """Symbolize a collection while preserving recording boundaries."""
        return [self.transform(item) for item in series]

    __call__ = transform
