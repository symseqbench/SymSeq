# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""
Preset grammar definitions, grouped by formalism.

Each submodule holds pure *data* — dicts of constructor kwargs for a generator
class — keeping grammar definitions separate from the grammar engines that
consume them. Engines depend on this package; this package depends on nothing,
so the relationship stays acyclic.

Submodules
----------
ag
    Presets for :class:`~symseq.generators.ag.artificial_grammar.ArtificialGrammar`
    (probabilistic finite-state automata, i.e. regular grammars).
cfg
    Presets for :class:`~symseq.generators.cfg.CFGGenerator`
    (context-free grammars in NLTK PCFG notation).
"""

from symseq.generators.presets import ag, cfg

__all__ = ["ag", "cfg"]
