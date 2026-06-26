# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Command-line interface for generating and inspecting SymSeq trial sets."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import typer
import yaml

from symseq.config import load_trial_set
from symseq.generators.registry import registered_names
from symseq.trial import Trial
from symseq.trial_set import TrialSet

app = typer.Typer(no_args_is_help=True)


def main() -> None:
    """Run the SymSeq command-line app."""
    app()


@app.command()
def generate(config: Path) -> None:
    """Generate a symbolic TrialSet and write it to symseq.storage.path."""
    raw_config = _load_raw_config(config)
    storage_path = _storage_path(raw_config)
    trial_set = load_trial_set(config)

    storage_path.mkdir(parents=True, exist_ok=True)
    _write_yaml(storage_path / "config.yaml", raw_config)
    _write_yaml(storage_path / "manifest.yaml", _manifest(raw_config, trial_set))
    _write_jsonl(storage_path / "trials.jsonl", _trial_records(trial_set))

    splits_dir = storage_path / "splits"
    splits_dir.mkdir(exist_ok=True)
    for split_name, indexes in trial_set.splits.items():
        records = (_trial_record(index, trial_set.trials[index]) for index in indexes)
        _write_jsonl(splits_dir / f"{split_name}.jsonl", records)

    typer.echo(f"Wrote {len(trial_set)} trials to {storage_path}")


@app.command()
def inspect(config: Path, head: int = typer.Option(5, min=1)) -> None:
    """Print a small preview of generated trials without writing files."""
    raw_config = _load_raw_config(config)
    _storage_path(raw_config)
    trial_set = load_trial_set(config)

    for record in _trial_records(trial_set, limit=head):
        typer.echo(json.dumps(record, sort_keys=True))


@app.command()
def validate(config: Path) -> None:
    """Validate that a config can be used by the SymSeq CLI."""
    raw_config = _load_raw_config(config)
    storage_path = _storage_path(raw_config)
    trial_set = load_trial_set(config)
    typer.echo(f"Valid config: {config}")
    typer.echo(f"Storage path: {storage_path}")
    typer.echo(f"Trials: {len(trial_set)}")


@app.command("list-generators")
def list_generators() -> None:
    """List registered symbolic sequence generators."""
    for name in registered_names():
        typer.echo(name)


def _load_raw_config(path: Path) -> dict[str, Any]:
    try:
        with open(path) as f:
            raw = yaml.safe_load(f)
    except Exception as exc:
        raise typer.BadParameter(str(exc), param_hint="CONFIG") from exc

    if raw is None:
        raise typer.BadParameter(f"config file {path} is empty", param_hint="CONFIG")
    if not isinstance(raw, dict):
        raise typer.BadParameter("config must contain a mapping at the top level", param_hint="CONFIG")
    return raw


def _storage_path(raw_config: dict[str, Any]) -> Path:
    try:
        storage = raw_config["symseq"]["storage"]
        storage_path = storage["path"]
    except KeyError as exc:
        raise typer.BadParameter(
            "config must define symseq.storage.path",
            param_hint="CONFIG",
        ) from exc

    if not storage_path:
        raise typer.BadParameter("symseq.storage.path must not be empty", param_hint="CONFIG")
    return Path(storage_path).expanduser()


def _manifest(raw_config: dict[str, Any], trial_set: TrialSet) -> dict[str, Any]:
    symseq_cfg = raw_config["symseq"]
    generator_cfg = symseq_cfg["generator"]
    return {
        "alphabet": _jsonable(trial_set.alphabet),
        "generator": {
            "type": generator_cfg["type"],
            "mode": generator_cfg.get("mode"),
            "preset": generator_cfg.get("preset"),
        },
        "seed": symseq_cfg.get("seed"),
        "splits": {name: len(indexes) for name, indexes in trial_set.splits.items()},
        "n_trials": len(trial_set),
    }


def _trial_records(trial_set: TrialSet, limit: int | None = None):
    for index, trial in enumerate(trial_set.trials):
        if limit is not None and index >= limit:
            break
        yield _trial_record(index, trial)


def _trial_record(index: int, trial: Trial) -> dict[str, Any]:
    return {
        "index": index,
        "symbols": _jsonable(trial.symbols),
        "states": _jsonable(trial.states),
        "targets": {
            name: {
                "values": _jsonable(target.values),
                "mask": _jsonable(target.mask),
                "kind": target.kind,
            }
            for name, target in trial.targets.items()
        },
        "meta": _jsonable(trial.meta),
    }


def _write_jsonl(path: Path, records) -> None:
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(_jsonable(data), f, sort_keys=False)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    main()
