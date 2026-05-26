# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Symbolic-sequence generators. Importing this package eagerly registers all
built-in generators with the registry so ``build(name, ...)`` works without
needing the caller to import each module by hand.
"""

from symseq.generators.ag import ArtificialGrammar  # registers "ArtificialGrammar"
from symseq.generators.dyck import DyckGenerator  # registers "Dyck"
from symseq.generators.nax import nAX  # noqa: F401 (registered if @register is added)
from symseq.generators.nback import NBack  # registers "NBack"
from symseq.generators.nad import NonAdjacentDependencies  # noqa: F401

__all__ = [
    "ArtificialGrammar",
    "DyckGenerator",
    "nAX",
    "NBack",
    "NonAdjacentDependencies",
]
