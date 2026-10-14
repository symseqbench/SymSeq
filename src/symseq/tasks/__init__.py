# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Task definitions. Importing this package eagerly registers all built-in
tasks so :func:`symseq.tasks.registry.build` works without per-class imports.
"""

from symseq.tasks.base import Task
from symseq.tasks.chunk import NGramChunk
from symseq.tasks.intrinsic import (
    Grammaticality,
    NAXIsTarget,
    NAXLabel,
    NBackMatch,
    NBackRole,
    PairIndex,
)
from symseq.tasks.materialize import (
    ConfiguredTrialSource,
    build_tasks,
    coerce_task_entries,
    materialize_targets,
    validate_task_compatibility,
)
from symseq.tasks.shift import NStepMemory, NStepPrediction

__all__ = [
    "ConfiguredTrialSource",
    "Grammaticality",
    "NAXIsTarget",
    "NAXLabel",
    "NBackMatch",
    "NBackRole",
    "NGramChunk",
    "NStepMemory",
    "NStepPrediction",
    "PairIndex",
    "Task",
    "build_tasks",
    "coerce_task_entries",
    "materialize_targets",
    "validate_task_compatibility",
]
