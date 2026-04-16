# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""
SymSeq Metrics: Hierarchical sequence complexity metrics.

This module provides a comprehensive suite of metrics for analyzing symbolic sequences
at multiple levels of abstraction:

**Level 1 - Token**: Individual symbol statistics
    - Token frequency and counts
    - Token duration statistics (frame-based and explicit)

**Level 2 - String**: Single sequence complexity
    - Entropy measures (Shannon, block, rate, EMC)
    - Compression-based complexity (gzip, LZW)
    - Linguistic complexity (n-gram diversity)
    - Regularity measures (permutation entropy)

**Level 3 - String-set**: Multi-sequence relationships
    - Distance metrics (Hamming, edit, NCD)
    - Information theory (mutual information, NMI)
    - Associative chunk strength (ACS variants)
    - String-set entropy

**Level 4 - Grammar**: Generative structure inference
    - Markov models (order selection, VLMC)
    - Hierarchical structure (MI decay analysis)
    - Chomsky hierarchy classification
    - Grammar complexity measures
    - CFG parsing (CYK algorithm)

All legacy functions are re-exported for backward compatibility.
Use submodules (token, string, stringset, grammar) for new code.
"""

from . import grammar, string, stringset, token
from .grammar.topological import topological_entropy
from .string.compression import compressibility, lzw_complexity
from .string.entropy import block_entropy, emc, entropy, entropy_rate
from .stringset.chunk_strength import (
    acs_bailey2008,
    anchor_acs_knowlton96,
    global_acs_knowlton96,
)
from .stringset.distances import (
    average_string_similarity,
    edit_distance,
    hamming_distance,
    pairwise_distances,
    string_similarity,
)

__version__ = "0.0.1"

__all__ = [
    "entropy",
    "block_entropy",
    "entropy_rate",
    "emc",
    "compressibility",
    "lzw_complexity",
    "hamming_distance",
    "edit_distance",
    "pairwise_distances",
    "string_similarity",
    "average_string_similarity",
    "topological_entropy",
    # "count",
    # "most_common",
    "global_acs_knowlton96",
    "anchor_acs_knowlton96",
    "acs_bailey2008",
    "token",
    "string",
    "stringset",
    "grammar",
    "__version__",
]
