# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Symbolic-sequence generators. Importing this package eagerly registers all
built-in generators with the registry so ``build(name, ...)`` works without
needing the caller to import each module by hand.
"""

from symseq.generators.ag import ArtificialGrammar  # registers "ArtificialGrammar"
from symseq.generators.cfg import CFGGenerator  # registers "CFG"
from symseq.generators.dyck import DyckGenerator  # registers "Dyck"
from symseq.generators.nad import (
    CrossedNonAdjacentDependencies,
    NestedNonAdjacentDependencies,
    NonAdjacentDependencies,
)
from symseq.generators.nax import nAX
from symseq.generators.nback import NBack  # registers "NBack"

__all__ = [
    "ArtificialGrammar",
    "CFGGenerator",
    "CrossedNonAdjacentDependencies",
    "DyckGenerator",
    "NBack",
    "NestedNonAdjacentDependencies",
    "NonAdjacentDependencies",
    "nAX",
]
