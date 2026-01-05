# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""
String-level metrics.

This module provides metrics that quantify the complexity of individual strings
or concatenated string sets.
"""

from .entropy import (
    entropy, 
    block_entropy, 
    entropy_rate, 
    emc, 
    detect_saturation_point,
    compute_adaptive_max_k,
    aggregate_saturation_results,
    compute_entropy_metrics_ensemble
)
from .compression import compressibility, lzw_complexity, compute_compression_metrics_ensemble
from .linguistic import linguistic_complexity
from .regularity import permutation_entropy

__all__ = [
    'entropy',
    'block_entropy',
    'entropy_rate',
    'emc',
    'detect_saturation_point',
    'compute_adaptive_max_k',
    'aggregate_saturation_results',
    'compute_entropy_metrics_ensemble',
    'compressibility',
    'lzw_complexity',
    'compute_compression_metrics_ensemble',
    'linguistic_complexity',
    'permutation_entropy',
]
