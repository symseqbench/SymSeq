# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""
TrialSource Protocol — the cross-library contract between symseq and seqbench.

Anything that provides Trials (a live generator producing fresh data, or a
pregenerated TrialSet replaying snapshots) satisfies this Protocol. Consumers
(e.g. seqbench) depend on this Protocol, not on symseq's class hierarchy.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from symseq.trial import Trial


@runtime_checkable
class TrialSource(Protocol):
    @property
    def alphabet(self) -> list[str]: ...

    def draw_trial(self) -> Trial: ...

    def draw_batch(self, n: int) -> list[Trial]: ...
