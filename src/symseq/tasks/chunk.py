# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""
N-gram chunk task: at position i, the target is the k-gram ending at position i
(i.e., the tuple ``(symbols[i-k+1], ..., symbols[i])``). Positions ``[0..k-2]``
are masked because no full k-gram is available.

This replaces the recognition part of the legacy ``chunk_targets`` function.
For "memory" or "prediction" of chunks, compose with :class:`NStepMemory` /
:class:`NStepPrediction` after attaching this task's target (or apply them to a
chunked view of the sequence).
"""

from __future__ import annotations

from symseq.tasks.base import Task
from symseq.tasks.registry import register
from symseq.trial import Target, Trial


@register("NGramChunk")
class NGramChunk(Task):
    """Per-token target: the k-gram (tuple of k symbols) ending at position i.

    Parameters
    ----------
    n : int
        Chunk size. Must be ``>= 1``. ``n=1`` is the identity chunking
        (each target is the singleton ``(symbols[i],)``).
    """

    def __init__(self, n: int):
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"NGramChunk.n must be a positive int, got {n!r}")
        self.n = n
        self.name = f"{n}_gram_chunk"

    def __call__(self, trial: Trial) -> Target:
        symbols = trial.symbols
        L = len(symbols)
        values: list = [None] * L
        mask: list[bool] = [False] * L
        if self.n > L:
            return Target(values=values, mask=mask, kind="per_token")
        for i in range(self.n - 1, L):
            values[i] = tuple(symbols[i - self.n + 1 : i + 1])
            mask[i] = True
        return Target(values=values, mask=mask, kind="per_token")
