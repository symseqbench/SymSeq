# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""
Task ABC — base class for everything that derives a Target from a Trial.

A Task is "what the model is being trained to do." Given a Trial, a Task
produces a Target. The configured task ``id`` determines the key under which
that Target is attached; task implementations do not own target identities.

Generic tasks (e.g. n-step prediction, n-gram chunking) work on any Trial.
Generator-specific tasks (e.g. Dyck max-depth, NBack role) can also live here
and may read ``trial.meta`` for paradigm-private context.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from symseq.trial import Target, Trial


class Task(ABC):
    """Abstract base for tasks."""

    granularity: str

    @abstractmethod
    def __call__(self, trial: Trial) -> Target: ...
