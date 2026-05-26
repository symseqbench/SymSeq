# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""
TrialSet — an offline collection of Trials with named splits.

Also implements the TrialSource Protocol via ``draw_trial`` / ``draw_batch`` so
the same downstream code can consume a live generator or a pregenerated snapshot
through one contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from symseq.trial import Trial


@dataclass
class TrialSet:
    """Collection of Trials with named index-based splits.

    Parameters
    ----------
    trials : list[Trial]
        All trials in this set, in canonical order.
    splits : dict[str, list[int]]
        Named splits. Each value is a list of indices into ``trials``.
    meta : dict[str, Any]
        Free-form metadata. Conventional keys: ``alphabet``, ``generator``,
        ``config``, ``seed``, ``version``.
    """

    trials: list[Trial]
    splits: dict[str, list[int]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    # Private cursor for source-style iteration. Not user-facing.
    _cursor: int = field(default=0, init=False, repr=False, compare=False)
    _draw_split: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Validate split indices against the trials list.
        n = len(self.trials)
        for name, idxs in self.splits.items():
            for i in idxs:
                if not (0 <= i < n):
                    raise IndexError(
                        f"split {name!r} contains out-of-range index {i} "
                        f"(TrialSet has {n} trials)"
                    )
        # Default the draw source to "train" if present.
        if "train" in self.splits:
            self._draw_split = "train"

    # ---------- collection interface ----------

    def __len__(self) -> int:
        return len(self.trials)

    def __iter__(self) -> Iterator[Trial]:
        return iter(self.trials)

    def __getitem__(self, idx: int) -> Trial:
        return self.trials[idx]

    def split(self, name: str) -> list[Trial]:
        """Return the trials belonging to a named split."""
        if name not in self.splits:
            raise KeyError(f"unknown split {name!r}; available: {sorted(self.splits)}")
        return [self.trials[i] for i in self.splits[name]]

    # ---------- TrialSource protocol ----------

    @property
    def alphabet(self) -> list[str]:
        return self.meta.get("alphabet", [])

    def draw_trial(self) -> Trial:
        """Return the next Trial in cursor order from the configured draw split.

        Cycles deterministically through the split. Defaults to ``"train"`` if
        present at construction time, otherwise iterates over all trials.
        """
        if self._draw_split is not None:
            idxs = self.splits[self._draw_split]
            if not idxs:
                raise IndexError(f"split {self._draw_split!r} is empty")
            i = idxs[self._cursor % len(idxs)]
        else:
            if not self.trials:
                raise IndexError("TrialSet is empty")
            i = self._cursor % len(self.trials)
        self._cursor += 1
        return self.trials[i]

    def draw_batch(self, n: int) -> list[Trial]:
        return [self.draw_trial() for _ in range(n)]

    def set_draw_split(self, name: str | None) -> None:
        """Configure which split ``draw_trial`` draws from. ``None`` = all trials."""
        if name is not None and name not in self.splits:
            raise KeyError(f"unknown split {name!r}; available: {sorted(self.splits)}")
        self._draw_split = name
        self._cursor = 0

    def reset_cursor(self) -> None:
        """Restart cursor-based drawing from the beginning of the current split."""
        self._cursor = 0
