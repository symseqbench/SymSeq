# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Compression-based complexity metrics."""

import warnings
from gzip import compress

import numpy as np
from joblib import Parallel, delayed

_GZIP_FRAMING_BYTES = len(compress(b""))


def gzip_payload_size(data: bytes) -> int:
    """
    Size in bytes of the gzip-compressed `data`, excluding the fixed gzip framing.

    A gzip member carries a 10-byte header, an 8-byte trailer and an empty final
    DEFLATE block, which together are constant (20 bytes) and independent of the
    content. For strings of only a few symbols this framing is larger than the
    content itself, so it is subtracted to leave the size of the compressed payload.

    Parameters
    ----------
    data : bytes
        Data to compress.

    Returns
    -------
    int
        ``len(gzip.compress(data))`` minus the size of ``gzip.compress(b"")``.
    """
    return len(compress(data)) - _GZIP_FRAMING_BYTES


def compressibility(sequence: list[str]) -> float:
    """
    Determine the compressibility ratio of a sequence using the DEFLATE (gzip) algorithm.

    Parameters
    ----------
    sequence : list of str
        Symbolic sequence (list of symbols).

    Returns
    -------
    float
        Compressibility ratio C_gzip(S) = |C(S)| / |S|, where |C(S)| is the gzip
        payload size in bytes (fixed gzip header and trailer excluded, see
        `gzip_payload_size`).

    Notes
    -----
    Properties:
    - C_gzip(S) > 0 for non-empty S; approximately <= 1 for incompressible data,
      and may exceed 1 slightly for short, high-entropy strings since DEFLATE can
      expand them by a few bits
    - Lower values indicate higher compressibility (more structure/redundancy)
    - Approximates Kolmogorov complexity for practical sequences
    - Universal: works for any alphabet
    - Resolution is one byte, so values for very short sequences (< ~10 symbols)
      are coarse
    """
    return gzip_payload_size("".join(sequence).encode()) / len(sequence)


def lzw_complexity(sequence: list[str], normalized: bool = True) -> float:
    """
    Determine the raw or normalized LZW complexity of a sequence.

    Parameters
    ----------
    sequence : list of str
        Symbolic sequence (list of symbols).
    normalized : bool, default=True
        Whether to normalize the LZW complexity by the length of the sequence.

    Returns
    -------
    float
        LZW complexity of the sequence.

    Notes
    -----
    The LZW complexity C_LZW(S) counts the number of codes emitted when
    encoding S, starting from a dictionary of its single symbols:
    C_LZW(S) = |codes(S)| / |S| (if normalized)

    Symbols are treated as atomic, so any hashable symbol (multi-character,
    non-Latin-1, ...) is supported and ``['A', 'BC']`` differs from ``['AB', 'C']``.

    Properties:
    - Normalized: 0 < C_LZW(S) <= 1
    - Related to the growth rate of pattern vocabulary
    - Approximates algorithmic complexity
    - Asymptotically optimal for stationary ergodic sources
    """
    dictionary = {(symbol,): i for i, symbol in enumerate(dict.fromkeys(sequence))}
    dict_size = len(dictionary)
    w = ()
    n_codes = 0

    for c in sequence:
        wc = (*w, c)
        if wc in dictionary:
            w = wc
        else:
            n_codes += 1
            dictionary[wc] = dict_size
            dict_size += 1
            w = (c,)

    if w:
        n_codes += 1

    if normalized:
        return n_codes / len(sequence)
    else:
        return n_codes


def _compute_compression_single(seq):
    """
    Compute compression metrics for a single sequence (shared by serial and parallel paths).

    Returns
    -------
    tuple
        (compressibility, lzw_complexity, errors). A metric that fails is NaN and its
        exception is described in `errors`, so results stay aligned with the input.
    """
    values = []
    errors = []
    for name, func in (("compressibility", compressibility), ("lzw_complexity", lzw_complexity)):
        try:
            values.append(func(seq))
        except Exception as e:
            values.append(np.nan)
            errors.append(f"{name}: {e!r}")
    return values[0], values[1], errors


def compute_compression_metrics_ensemble(
    sequences: list[list[str]], min_sequence_length: int = 10, n_jobs: int = 1
) -> dict:
    """
    Compute compression-based metrics on an ensemble of sequences.

    Parameters
    ----------
    sequences : list of list of str
        List of symbolic sequences.
    min_sequence_length : int, default=10
        Minimum sequence length to include in analysis.
    n_jobs : int, default=1
        Number of parallel jobs. -1 means using all processors.
        Set to 1 for sequential processing (default for backward compatibility).

    Returns
    -------
    dict with keys:
        'compressibility': list of float
        'lzw_complexity': list of float
        'n_sequences_analyzed': int

    Notes
    -----
    Both lists are index-aligned to the sequences with ``len >= min_sequence_length``. A metric that
    fails for a sequence is NaN at that position and a warning summarizes the failures; aggregate
    with ``np.nanmean`` or similar.
    """
    # Filter sequences by minimum length
    valid_sequences = [seq for seq in sequences if len(seq) >= min_sequence_length]

    if not valid_sequences:
        return {"compressibility": [], "lzw_complexity": [], "n_sequences_analyzed": 0}

    # Compute metrics for each sequence
    if n_jobs != 1:
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_compute_compression_single)(seq) for seq in valid_sequences
        )
    else:
        results = [_compute_compression_single(seq) for seq in valid_sequences]

    compressibilities = [r[0] for r in results]
    lzw_complexities = [r[1] for r in results]

    failures = [(i, r[2]) for i, r in enumerate(results) if r[2]]
    if failures:
        warnings.warn(
            f"Compression metrics failed for {len(failures)} of {len(valid_sequences)} sequences "
            f"(NaN returned at those positions). First failure, sequence {failures[0][0]}: "
            f"{'; '.join(failures[0][1])}",
            UserWarning,
            stacklevel=2,
        )

    return {
        "compressibility": compressibilities,
        "lzw_complexity": lzw_complexities,
        "n_sequences_analyzed": len(valid_sequences),
    }
