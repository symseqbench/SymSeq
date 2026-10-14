# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Regular-grammar inference backends for :class:`ArtificialGrammar`."""

from .markov import infer_markov
from .vlmc import infer_vlmc

__all__ = ["infer_markov", "infer_vlmc"]
