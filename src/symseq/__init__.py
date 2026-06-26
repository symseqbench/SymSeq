# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Top-level package for symseq."""

import importlib
from typing import TYPE_CHECKING

from .trial import Trial, Target
from .trial_set import TrialSet
from .trial_source import TrialSource

if TYPE_CHECKING:
    from . import generators, tasks
    from .config import load_trial_set
    from .tasks.base import Task

__author__ = """Barna Zajzon"""
__email__ = 'barna.zajzon@gmail.com'
__version__ = '0.1.0'

__all__ = [
    "Target",
    "Task",
    "Trial",
    "TrialSet",
    "TrialSource",
    "generators",
    "load_trial_set",
    "tasks",
]


def __getattr__(name: str):
    if name == "generators":
        return importlib.import_module(f"{__name__}.generators")
    if name == "tasks":
        return importlib.import_module(f"{__name__}.tasks")
    if name == "load_trial_set":
        from .config import load_trial_set

        return load_trial_set
    if name == "Task":
        from .tasks.base import Task

        return Task
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
