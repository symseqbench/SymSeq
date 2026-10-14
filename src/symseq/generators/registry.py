# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""
Generator registry — replaces the hardcoded if/elif dispatch in SeqWrapper.

Each generator class registers itself via @register("Name"); the config loader
looks up the class by name and instantiates it.
"""

from __future__ import annotations

from typing import Type

from symseq.core.sequencer import SymbolicSequencer


_REGISTRY: dict[str, Type[SymbolicSequencer]] = {}


def register(name: str):
    """Class decorator that registers a generator under `name`."""

    def _decorator(cls: Type[SymbolicSequencer]) -> Type[SymbolicSequencer]:
        if name in _REGISTRY:
            raise ValueError(f"Generator {name!r} is already registered to {_REGISTRY[name].__name__}.")
        _REGISTRY[name] = cls
        return cls

    return _decorator


def build(name: str, **params) -> SymbolicSequencer:
    """Instantiate a registered generator by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown generator {name!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**params)


def registered_names() -> list[str]:
    return sorted(_REGISTRY)
