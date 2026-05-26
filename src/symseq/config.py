# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""
Config loader for building TrialSets declaratively.

Replaces the SeqWrapper config-parsing role with a single free function:
``load_trial_set(source) -> TrialSet``. Schema is intentionally minimal and
validated by hand (no pydantic dependency).

Schema (YAML or TOML)::

    symseq:
      seed: 42                       # optional int; sets the generator's RNG
      generator:
        type: NBack                  # required; must be a registered generator
        params:                      # passed to the registered class __init__
          n: 2
          alphabet_size: 8
      trial_set:
        n_trials: 1200               # required, total trials to generate
        splits:                      # optional dict of name -> int (count) or float (fraction)
          train: 1000
          test: 200
        gen_params: {}               # optional kwargs forwarded to each generate_trial call

Convenience for ArtificialGrammar::

      generator:
        type: ArtificialGrammar
        preset: Elman                # uses ArtificialGrammar.from_preset(...)
        seed: 42
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

import symseq.generators  # noqa: F401 — eagerly register built-in generators
from symseq.generators.ag import ArtificialGrammar
from symseq.generators.registry import build as build_generator
from symseq.trial_set import TrialSet


def load_trial_set(source: str | Path | dict) -> TrialSet:
    """Load a config and produce a TrialSet.

    Parameters
    ----------
    source
        Either a path to a YAML/TOML file, or an already-parsed config dict.

    Returns
    -------
    TrialSet
        Populated with the generated trials, named splits, and a meta dict
        carrying the resolved config snapshot, seed, alphabet, and a reference
        to the live generator (for downstream online use).
    """
    cfg = _load_config(source)
    _validate(cfg)

    seed = cfg.get("seed")
    gen_cfg = cfg["generator"]
    generator = _build_generator(gen_cfg, seed=seed)

    ts_cfg = cfg["trial_set"]
    n_trials = int(ts_cfg["n_trials"])
    gen_params = ts_cfg.get("gen_params") or {}

    trials = generator.generate_trials(n=n_trials, **gen_params)

    splits = _resolve_splits(n_trials, ts_cfg.get("splits") or {})

    meta = {
        "config": cfg,
        "seed": seed,
        "alphabet": list(generator.alphabet),
        "generator": generator,
    }
    return TrialSet(trials=trials, splits=splits, meta=meta)


# ----------------------------- internals --------------------------------------


def _load_config(source: str | Path | dict) -> dict:
    if isinstance(source, dict):
        cfg = source
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"config file not found: {path}")
        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            import yaml
            with open(path) as f:
                cfg = yaml.safe_load(f)
        elif suffix == ".toml":
            import toml
            with open(path) as f:
                cfg = toml.load(f)
        else:
            raise ValueError(f"unsupported config extension {suffix!r}; use .yaml/.yml/.toml")

    if "symseq" not in cfg:
        raise ValueError("config missing top-level 'symseq' section")
    return cfg["symseq"]


def _validate(cfg: dict) -> None:
    if "generator" not in cfg:
        raise ValueError("config.symseq missing 'generator' section")
    if "type" not in cfg["generator"]:
        raise ValueError("config.symseq.generator missing 'type'")
    if "trial_set" not in cfg:
        raise ValueError("config.symseq missing 'trial_set' section")
    if "n_trials" not in cfg["trial_set"]:
        raise ValueError("config.symseq.trial_set missing 'n_trials'")


def _build_generator(gen_cfg: dict, seed: int | None) -> Any:
    type_name = gen_cfg["type"]
    params = dict(gen_cfg.get("params") or {})

    # Inject seed -> rng for generators that accept either an rng or seed kwarg.
    # We pass an rng to be consistent across all generators.
    if seed is not None and "rng" not in params and "seed" not in params:
        params["rng"] = np.random.default_rng(seed)

    # Special case: ArtificialGrammar preset constructor.
    if type_name == "ArtificialGrammar" and "preset" in gen_cfg:
        return ArtificialGrammar.from_preset(
            preset_name=gen_cfg["preset"],
            seed=seed if seed is not None else 42,
        )

    return build_generator(type_name, **params)


def _resolve_splits(
    n_total: int, splits_cfg: dict[str, int | float]
) -> dict[str, list[int]]:
    """Turn ``{"train": 1000, "test": 200}`` or fractional variants into index lists.

    Splits are allocated in declaration order, consuming the trial list from the
    front. Counts and fractions can be mixed; fractions are interpreted as a
    fraction of ``n_total``. Total allocation must not exceed ``n_total``.
    """
    out: dict[str, list[int]] = {}
    cursor = 0
    for name, value in splits_cfg.items():
        if isinstance(value, float) and 0.0 < value <= 1.0:
            size = int(round(value * n_total))
        elif isinstance(value, int) and value >= 0:
            size = value
        else:
            raise ValueError(
                f"split {name!r}: value must be int>=0 or float in (0,1], got {value!r}"
            )
        end = cursor + size
        if end > n_total:
            raise ValueError(
                f"splits exceed n_trials={n_total}: {name!r} would extend to index {end}"
            )
        out[name] = list(range(cursor, end))
        cursor = end
    return out
