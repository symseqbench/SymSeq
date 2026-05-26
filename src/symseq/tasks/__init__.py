# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Task definitions. Importing this package eagerly registers all built-in
tasks so :func:`symseq.tasks.registry.build` works without per-class imports.
"""

from symseq.tasks.base import Task
from symseq.tasks.chunk import NGramChunk
from symseq.tasks.shift import NStepMemory, NStepPrediction

__all__ = [
    "Task",
    "NStepMemory",
    "NStepPrediction",
    "NGramChunk",
]
