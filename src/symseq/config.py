# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""
Config loader for building TrialSets declaratively.

Replaces the SeqWrapper config-parsing role with a single free function:
``load_trial_set(source) -> TrialSet``. Schema is intentionally minimal and
validated by hand (no pydantic dependency).

Schema (YAML)::

    symseq:
      seed: 42                       # optional int; overrides run.seed
      generator:
        type: NBack                  # required; must be a registered generator
        params:                      # passed to the registered class __init__
          n: 2
          alphabet_size: 8
        trial_params: {}             # optional kwargs forwarded to generate_trial
      tasks:                          # optional configured target materializers
        - id: next_token             # unique public target key
          type: NStepPrediction      # registered task implementation
          params:                    # optional constructor arguments
            n: 1
      trial_set:
        n_trials: 1200               # required, total trials to generate
        splits:                      # optional dict of name -> int (count) or float (fraction)
          train: 1000
          test: 200

Convenience for ArtificialGrammar::

      generator:
        type: ArtificialGrammar
        preset: Elman                # uses ArtificialGrammar.from_preset(...)
        seed: 42

      generator:
        type: ArtificialGrammar
        mode: random                 # uses ArtificialGrammar.from_constraints(...)
        params: {}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import symseq.generators  # noqa: F401 — eagerly register built-in generators
from symseq.generators.ag import ArtificialGrammar
from symseq.generators.registry import build as build_generator
from symseq.tasks.materialize import (
    ConfiguredTrialSource,
    build_tasks,
    materialize_targets,
)
from symseq.trial_set import TrialSet


def load_trial_set(source: str | Path | dict) -> TrialSet:
    """Load a config and produce a TrialSet.

    Parameters
    ----------
    source
        Either a path to a YAML file, or an already-parsed config dict.

    Returns
    -------
    TrialSet
        Populated with the generated trials, named splits, and a meta dict
        carrying the resolved config snapshot, seed, alphabet, and a reference
        to the live generator (for downstream online use).
    """
    cfg = _load_config(source)
    _validate(cfg)
    tasks = build_tasks(cfg.get("tasks"), where="config.symseq.tasks")

    seed = cfg.get("seed")
    gen_cfg = cfg["generator"]
    generator = _build_generator(gen_cfg, seed=seed)
    live_generator = ConfiguredTrialSource(generator, tasks)

    ts_cfg = cfg["trial_set"]
    n_trials = int(ts_cfg["n_trials"])
    trial_params = resolve_trial_params(cfg)

    trials = generator.generate_trials(n=n_trials, **trial_params)
    for trial in trials:
        materialize_targets(trial, tasks)

    splits = _resolve_splits(n_trials, ts_cfg.get("splits") or {})

    meta = {
        "config": cfg,
        "seed": seed,
        "alphabet": list(generator.alphabet),
        "generator": live_generator,
        "task_ids": list(tasks),
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
        else:
            raise ValueError(f"unsupported config extension {suffix!r}; use .yaml/.yml")

    cfg = _migrate_config(cfg)
    if "symseq" not in cfg:
        raise ValueError("config missing top-level 'symseq' section")
    symseq_cfg = cfg["symseq"]
    if symseq_cfg.get("seed") is None and cfg.get("run", {}).get("seed") is not None:
        symseq_cfg["seed"] = cfg["run"]["seed"]
    if "generator" in symseq_cfg:
        _apply_symbol_space_defaults(symseq_cfg["generator"], cfg.get("symbol_space"))
    return symseq_cfg


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

    # Special case: ArtificialGrammar preset constructor.
    if type_name == "ArtificialGrammar" and "preset" in gen_cfg:
        return ArtificialGrammar.from_preset(
            preset_name=gen_cfg["preset"],
            seed=seed if seed is not None else 42,
        )

    # Special case: random ArtificialGrammar constructor.
    if type_name == "ArtificialGrammar" and gen_cfg.get("mode") == "random":
        if seed is not None and "rng" not in params and "seed" not in params:
            params["seed"] = seed
        return ArtificialGrammar.from_constraints(**params)

    # Special case: CFG preset constructor.
    if type_name == "CFG" and "preset" in gen_cfg:
        from symseq.generators.cfg import CFGGenerator

        return CFGGenerator.from_preset(
            preset_name=gen_cfg["preset"],
            seed=seed,
            **params,
        )

    # Inject seed -> rng for generators that accept either an rng or seed kwarg.
    # We pass an rng to be consistent across all generators.
    if seed is not None and "rng" not in params and "seed" not in params:
        params["rng"] = np.random.default_rng(seed)

    return build_generator(type_name, **params)


def _migrate_config(cfg: dict) -> dict:
    cfg = _deep_copy_config(cfg)
    if "dataset" in cfg:
        dataset = cfg.pop("dataset")
        cfg.setdefault("run", {})
        cfg["run"].setdefault("seed", dataset.get("seed"))
        alphabet = dict(dataset.get("alphabet") or {})
        if alphabet:
            eos = alphabet.pop("eos", "#")
            cfg.setdefault("symbol_space", {})
            cfg["symbol_space"].setdefault("alphabet", alphabet)
            cfg["symbol_space"].setdefault("eos", eos)
        if "trial_length" in dataset and cfg.get("symseq") is not None:
            length = dict(dataset["trial_length"])
            length.pop("distribution", None)
            cfg["symseq"].setdefault("trial_constraints", {})
            cfg["symseq"]["trial_constraints"].setdefault("length", length)

    symseq_cfg = cfg.get("symseq")
    if symseq_cfg is not None:
        gen_cfg = symseq_cfg.get("generator")
        ts_cfg = symseq_cfg.get("trial_set")
        if gen_cfg is not None and ts_cfg and "gen_params" in ts_cfg:
            gen_cfg.setdefault("trial_params", dict(ts_cfg.pop("gen_params") or {}))
    return cfg


def _deep_copy_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _deep_copy_config(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_copy_config(v) for v in value]
    return value


def _apply_symbol_space_defaults(gen_cfg: dict, symbol_space: dict | None) -> None:
    if not symbol_space:
        return
    params = gen_cfg.setdefault("params", {})
    alphabet = dict(symbol_space.get("alphabet") or {})
    if not alphabet:
        return
    type_name = gen_cfg.get("type")
    symbols = alphabet.get("symbols")
    size = alphabet.get("size")
    if type_name == "ArtificialGrammar" and gen_cfg.get("mode") == "random":
        if size is not None:
            params.setdefault("alphabet_size", size)
        if symbol_space.get("eos") is not None:
            params.setdefault("eos", symbol_space.get("eos"))
    elif type_name == "NBack":
        if symbols is not None:
            params.setdefault("alphabet", symbols)
        elif size is not None:
            params.setdefault("alphabet_size", size)


def resolve_trial_params(symseq_cfg: Any) -> dict[str, Any]:
    """Return per-trial kwargs with safe inferred length constraints applied.

    Explicit ``generator.trial_params`` always wins. Unsupported or ranged
    generator-specific length policies are left to the generator/source.
    Accepts either a plain dict or a typed config object with matching
    attributes, so downstream packages can delegate SymSeq trial policy here.
    """
    gen_cfg = _cfg_get(symseq_cfg, "generator")
    params = dict(_cfg_get(gen_cfg, "trial_params", {}) or {})
    constraints = _cfg_get(symseq_cfg, "trial_constraints")
    length = _cfg_get(constraints, "length") if constraints is not None else None
    if not length:
        return params

    gen_type = _cfg_get(gen_cfg, "type")
    min_len = int(_cfg_get(length, "min"))
    max_len = int(_cfg_get(length, "max"))
    if gen_type in {"ArtificialGrammar", "nAX"}:
        if not any(k in params for k in ("length_range", "min_length", "max_length")):
            params["length_range"] = [min_len, max_len]
    elif gen_type == "NBack":
        if "seq_length" not in params and min_len == max_len:
            params["seq_length"] = max_len
    elif gen_type == "NonAdjacentDependencies":
        if "filler_len" not in params and min_len == max_len:
            if max_len < 2:
                raise ValueError(
                    "NonAdjacentDependencies fixed trial length must be >= 2 "
                    f"to infer filler_len, got {max_len}"
                )
            params["filler_len"] = max_len - 2
    return params


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


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
            size = round(value * n_total)
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
