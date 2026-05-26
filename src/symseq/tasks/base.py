# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""
Task ABC — base class for everything that derives a Target from a Trial.

A Task is "what the model is being trained to do." Given a Trial (input + any
intrinsic targets the generator already populated), a Task produces a Target
that can be attached to ``trial.targets[task.name]``.

Generic tasks (e.g. n-step prediction, n-gram chunking) work on any Trial.
Generator-specific tasks (e.g. Dyck max-depth, NBack role) can also live here
and may read ``trial.meta`` for paradigm-private context.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from symseq.trial import Target, Trial


class Task(ABC):
    """Abstract base for tasks. Subclasses must set ``name`` and implement ``__call__``."""

    name: str

    @abstractmethod
    def __call__(self, trial: Trial) -> Target: ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
