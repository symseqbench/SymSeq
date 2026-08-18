# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""
Shift-based tasks: predict a token n steps back (memory) or n steps forward
(prediction). Pure functions of ``trial.symbols``; paradigm-agnostic.

Note on semantics: ``NStepMemory(n=1)`` means "at position i, the target is
``symbols[i-1]``". Positions where no valid target exists are masked. The
``n=k`` form generalises: first ``k`` positions are masked for memory; last
``k`` positions are masked for prediction. This is the natural semantics
and differs from the legacy ``memory_targets(seq, n_mem=0)`` which was offset
by one.
"""

from __future__ import annotations

from symseq.tasks.base import Task
from symseq.tasks.registry import register
from symseq.trial import Target, Trial


@register("NStepMemory")
class NStepMemory(Task):
    """Per-token target: at position i, predict ``symbols[i-n]``.

    Positions ``[0..n-1]`` are masked (no valid target exists).
    """

    granularity = "per_token"

    def __init__(self, n: int):
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"NStepMemory.n must be a positive int, got {n!r}")
        self.n = n

    def __call__(self, trial: Trial) -> Target:
        symbols = trial.symbols
        L = len(symbols)
        if self.n >= L:
            # No valid positions; everything masked.
            return Target(values=[None] * L, mask=[False] * L, granularity="per_token")
        values = [None] * self.n + list(symbols[: L - self.n])
        mask = [False] * self.n + [True] * (L - self.n)
        return Target(values=values, mask=mask, granularity="per_token")


@register("NStepPrediction")
class NStepPrediction(Task):
    """Per-token target: at position i, predict ``symbols[i+n]``.

    Last ``n`` positions are masked.
    """

    granularity = "per_token"

    def __init__(self, n: int):
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"NStepPrediction.n must be a positive int, got {n!r}")
        self.n = n

    def __call__(self, trial: Trial) -> Target:
        symbols = trial.symbols
        L = len(symbols)
        if self.n >= L:
            return Target(values=[None] * L, mask=[False] * L, granularity="per_token")
        values = list(symbols[self.n:]) + [None] * self.n
        mask = [True] * (L - self.n) + [False] * self.n
        return Target(values=values, mask=mask, granularity="per_token")
