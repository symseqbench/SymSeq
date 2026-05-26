# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Top-level package for symseq."""

from . import generators
from . import tasks
from .config import load_trial_set
from .tasks.base import Task
from .trial import Trial, Target
from .trial_set import TrialSet
from .trial_source import TrialSource

__author__ = """Barna Zajzon"""
__email__ = 'barna.zajzon@gmail.com'
__version__ = '0.1.0'
