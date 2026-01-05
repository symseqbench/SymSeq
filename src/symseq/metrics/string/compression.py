# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Compression-based complexity metrics."""

from gzip import compress
import numpy as np
from typing import List, Dict
from joblib import Parallel, delayed


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
        Compressibility ratio C_gzip(S) = |C(S)| / |S|.

    Notes
    -----
    Properties:
    - 0 < C_gzip(S) <= 1
    - Lower values indicate higher compressibility (more structure/redundancy)
    - Approximates Kolmogorov complexity for practical sequences
    - Universal: works for any alphabet
    """
    return len(compress("".join(sequence).encode())) / len(sequence)


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
    The LZW complexity C_LZW(S) counts the number of distinct patterns
    in the dictionary:
    C_LZW(S) = |D(S)| / |S| (if normalized)

    Properties:
    - Normalized: 0 < C_LZW(S) <= 1
    - Related to the growth rate of pattern vocabulary
    - Approximates algorithmic complexity
    - Asymptotically optimal for stationary ergodic sources
    """
    dict_size = 256
    dictionary = {chr(i): i for i in range(dict_size)}
    w = ""
    result = []

    for c in sequence:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            result.append(dictionary[w])
            dictionary[wc] = dict_size
            dict_size += 1
            w = c

    if w:
        result.append(dictionary[w])

    if normalized:
        return len(result) / len(sequence)
    else:
        return len(result)


def _compute_compression_single(seq):
    """Compute compression metrics for single sequence (helper for parallelization)."""
    try:
        comp = compressibility(seq)
        lzw = lzw_complexity(seq, normalized=True)
        return comp, lzw
    except Exception:
        return None, None


def compute_compression_metrics_ensemble(sequences: List[list[str]],
                                         min_sequence_length: int = 10,
                                         n_jobs: int = 1) -> Dict:
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
    """
    # Filter sequences by minimum length
    valid_sequences = [seq for seq in sequences if len(seq) >= min_sequence_length]
    
    if not valid_sequences:
        return {
            'compressibility': [],
            'lzw_complexity': [],
            'n_sequences_analyzed': 0
        }
    
    # Compute metrics for each sequence
    if n_jobs != 1:
        # Parallel computation
        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_compute_compression_single)(seq) for seq in valid_sequences
        )
        compressibilities = [r[0] for r in results if r[0] is not None]
        lzw_complexities = [r[1] for r in results if r[1] is not None]
    else:
        # Sequential computation (original behavior)
        compressibilities = []
        lzw_complexities = []
        
        for seq in valid_sequences:
            try:
                comp = compressibility(seq)
                compressibilities.append(comp)
            except Exception:
                pass
            
            try:
                lzw = lzw_complexity(seq, normalized=True)
                lzw_complexities.append(lzw)
            except Exception:
                pass
    
    return {
        'compressibility': compressibilities,
        'lzw_complexity': lzw_complexities,
        'n_sequences_analyzed': len(valid_sequences)
    }
