# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Explicit tasks backed by target candidates computed by generators."""

from __future__ import annotations

from symseq.tasks.base import Task
from symseq.tasks.registry import register
from symseq.trial import Target, Trial


class _IntrinsicTask(Task):
    intrinsic_id: str

    def __call__(self, trial: Trial) -> Target:
        try:
            target = trial.intrinsic_targets[self.intrinsic_id]
        except KeyError as exc:
            available = sorted(trial.intrinsic_targets)
            raise ValueError(
                f"{type(self).__name__} requires intrinsic target "
                f"{self.intrinsic_id!r}; available: {available or 'none'}"
            ) from exc
        if target.granularity != self.granularity:
            raise ValueError(
                f"{type(self).__name__} expects intrinsic target "
                f"{self.intrinsic_id!r} to have granularity "
                f"{self.granularity!r}; got {target.granularity!r}"
            )
        return Target(
            values=list(target.values) if isinstance(target.values, list) else target.values,
            mask=list(target.mask) if target.mask is not None else None,
            granularity=self.granularity,
        )


def _register_intrinsic_task(
    type_name: str,
    intrinsic_id: str,
    granularity: str,
) -> type[_IntrinsicTask]:
    """Create a named task adapter for one declared intrinsic target."""
    task_type = type(
        type_name,
        (_IntrinsicTask,),
        {
            "__module__": __name__,
            "intrinsic_id": intrinsic_id,
            "granularity": granularity,
        },
    )
    return register(type_name)(task_type)


Grammaticality = _register_intrinsic_task(
    "Grammaticality", "grammaticality", "per_trial"
)
PairIndex = _register_intrinsic_task("PairIndex", "pair_index", "per_trial")
NBackMatch = _register_intrinsic_task("NBackMatch", "nback_match", "per_token")
NBackRole = _register_intrinsic_task("NBackRole", "nback_role", "per_token")
NAXLabel = _register_intrinsic_task("NAXLabel", "nax_label", "per_trial")
NAXIsTarget = _register_intrinsic_task(
    "NAXIsTarget", "nax_is_target", "per_trial"
)
