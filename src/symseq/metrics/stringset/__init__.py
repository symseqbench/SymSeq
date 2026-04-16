# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""
String-set level metrics.

This module provides metrics that characterize relationships and complexity
across multiple strings.
"""

from .chunk_strength import (
    acs_bailey2008,
    anchor_acs_knowlton96,
    global_acs_knowlton96,
)
from .distances import (
    edit_distance,
    hamming_distance,
    normalized_compression_distance,
    pairwise_distances,
    pairwise_distances_parallel,
)
from .information import (
    mutual_information_strings,
    normalized_mutual_information,
    pairwise_mutual_information,
    pairwise_mutual_information_parallel,
    string_set_entropy,
)

__all__ = [
    'hamming_distance',
    'edit_distance',
    'normalized_compression_distance',
    'pairwise_distances',
    'pairwise_distances_parallel',
    'global_acs_knowlton96',
    'anchor_acs_knowlton96',
    'acs_bailey2008',
    'mutual_information_strings',
    'normalized_mutual_information',
    'pairwise_mutual_information',
    'pairwise_mutual_information_parallel',
    'string_set_entropy',
]
