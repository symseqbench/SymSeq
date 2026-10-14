# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""
Grammar-level metrics.

This module provides metrics that aim to infer the generative mechanisms
underlying observed sequences and characterize the computational capacity
and structural complexity of the underlying grammar.
"""

from .chomsky import chomsky_classification
from .complexity import grammar_rule_complexity, grammar_state_complexity
from .hierarchical import grassberger_entropy, mi_decay_analysis, mi_decay_analysis_parallel
from .inference import cyk_parse
from .markov import context_counts, markov_order_selection
from .topological import TopologicalEntropyWarning, topological_entropy
from .vlmc import vlmc_fit

__all__ = [
    "topological_entropy",
    "TopologicalEntropyWarning",
    "markov_order_selection",
    "context_counts",
    "vlmc_fit",
    "mi_decay_analysis",
    "mi_decay_analysis_parallel",
    "grassberger_entropy",
    "chomsky_classification",
    "grammar_rule_complexity",
    "grammar_state_complexity",
    "cyk_parse",
]
