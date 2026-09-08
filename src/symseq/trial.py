# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""
Trial and Target dataclasses — the atomic data unit produced by symseq generators.

A Trial is the output of a single generator.generate_trial() call: one full
generation with its observable symbols, optional state-indexed view, configured
targets, generator-intrinsic target candidates, and free-form metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class Target:
    values: list[Any] | Any
    mask: list[bool] | None
    granularity: Literal["per_token", "per_trial"]


@dataclass
class Trial:
    symbols: list[str]
    states: list[str] | None = None
    targets: dict[str, Target] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    intrinsic_targets: dict[str, Target] = field(default_factory=dict, repr=False)
