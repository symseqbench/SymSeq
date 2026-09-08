# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Build and apply configured SymSeq tasks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from symseq.tasks.base import Task
from symseq.trial import Trial


def coerce_task_entries(
    entries: Any,
    *,
    where: str = "configured SymSeq tasks",
) -> list[tuple[str, str, dict[str, Any]]]:
    """Validate task entries and return ``(id, type, params)`` tuples.

    Mapping entries use the strict public schema. Attribute-based entries are
    also accepted so typed configuration objects can use the same validation.
    """
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise ValueError(f"{where} must be a list")

    coerced: list[tuple[str, str, dict[str, Any]]] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        entry_where = f"{where}[{index}]"
        if isinstance(entry, Mapping):
            unknown = set(entry) - {"id", "type", "params"}
            if unknown:
                raise ValueError(f"{entry_where} has unknown keys {sorted(unknown)}")
            task_id = entry.get("id")
            type_name = entry.get("type")
            raw_params = entry.get("params", {})
        else:
            task_id = getattr(entry, "id", None)
            type_name = getattr(entry, "type", None)
            raw_params = getattr(entry, "params", {})

        if raw_params is None:
            raw_params = {}
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"{entry_where}.id must be a non-empty string, got {task_id!r}")
        if task_id in seen:
            raise ValueError(f"{where}[*].id must be unique; duplicate {task_id!r}")
        seen.add(task_id)
        if not isinstance(type_name, str) or not type_name:
            raise ValueError(f"{entry_where}.type must be a non-empty string, got {type_name!r}")
        if not isinstance(raw_params, Mapping):
            raise ValueError(f"{entry_where}.params must be a mapping, got {raw_params!r}")
        coerced.append((task_id, type_name, dict(raw_params)))
    return coerced


def build_tasks(
    entries: Any,
    *,
    where: str = "configured SymSeq tasks",
) -> dict[str, Task]:
    """Build configured tasks keyed by their validated output IDs."""
    from symseq.tasks import registry

    return {
        task_id: registry.build(type_name, **params)
        for task_id, type_name, params in coerce_task_entries(entries, where=where)
    }


def validate_task_compatibility(source: Any, tasks: Mapping[str, Task]) -> None:
    """Reject intrinsic tasks that the source cannot provide before drawing."""
    available = getattr(source, "intrinsic_target_granularities", {})
    for task_id, task in tasks.items():
        intrinsic_id = getattr(task, "intrinsic_id", None)
        if intrinsic_id is None:
            continue
        if intrinsic_id not in available:
            raise ValueError(
                f"task {task_id!r} ({type(task).__name__}) requires intrinsic target "
                f"{intrinsic_id!r}, but {type(source).__name__} provides "
                f"{sorted(available) or 'none'}"
            )
        source_granularity = available[intrinsic_id]
        if source_granularity != task.granularity:
            raise ValueError(
                f"task {task_id!r} ({type(task).__name__}) expects {intrinsic_id!r} "
                f"to be {task.granularity!r}, but {type(source).__name__} declares "
                f"{source_granularity!r}"
            )


def materialize_targets(trial: Trial, tasks: Mapping[str, Task]) -> None:
    """Replace a trial's public targets in place with configured task outputs."""
    trial.targets = {task_id: task(trial) for task_id, task in tasks.items()}


class ConfiguredTrialSource:
    """TrialSource wrapper that materializes configured tasks on every draw."""

    def __init__(self, source: Any, tasks: Mapping[str, Task]):
        validate_task_compatibility(source, tasks)
        self.source = source
        self.tasks = dict(tasks)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.source, name)

    @property
    def rng(self) -> Any:
        return self.source.rng

    @rng.setter
    def rng(self, value: Any) -> None:
        self.source.rng = value

    def draw_trial(self, **kwargs: Any) -> Trial:
        trial = self.source.draw_trial(**kwargs)
        materialize_targets(trial, self.tasks)
        return trial

    def draw_batch(self, n: int, **kwargs: Any) -> list[Trial]:
        trials = self.source.draw_batch(n, **kwargs)
        for trial in trials:
            materialize_targets(trial, self.tasks)
        return trials
