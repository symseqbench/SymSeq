# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""
Grammar-level metrics.

This module provides metrics that aim to infer the generative mechanisms
underlying observed sequences and characterize the computational capacity
and structural complexity of the underlying grammar.
"""

from .topological import topological_entropy
from .markov import markov_order_selection, vlmc_fit
from .hierarchical import mi_decay_analysis, mi_decay_analysis_parallel, grassberger_entropy
from .chomsky import chomsky_classification
from .complexity import grammar_rule_complexity, grammar_state_complexity
from .inference import cyk_parse

__all__ = [
    'topological_entropy',
    'markov_order_selection',
    'vlmc_fit',
    'mi_decay_analysis',
    'mi_decay_analysis_parallel',
    'grassberger_entropy',
    'chomsky_classification',
    'grammar_rule_complexity',
    'grammar_state_complexity',
    'cyk_parse',
]
