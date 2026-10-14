import json
import textwrap

import yaml
from typer.testing import CliRunner

from symseq.cli import app

runner = CliRunner()


def _random_ag_config(storage_path):
    return {
        "symseq": {
            "seed": 42,
            "storage": {"path": str(storage_path)},
            "generator": {
                "type": "ArtificialGrammar",
                "mode": "random",
                "params": {
                    "label": "Random AG CLI test",
                    "alphabet_size": 4,
                    "ambiguities": 1,
                    "ambiguity_depth": 2,
                    "n_start_states": 1,
                    "n_terminal_states": 1,
                    "transition_density": 0.35,
                    "assume_equiprobable": True,
                    "min_string_length": 2,
                    "verbose": False,
                },
                "trial_params": {"length_range": [2, 12]},
            },
            "trial_set": {
                "n_trials": 6,
                "splits": {"train": 4, "test": 2},
            },
        }
    }


def _write_yaml(path, data):
    path.write_text(yaml.safe_dump(data))


def test_list_generators_includes_artificial_grammar():
    result = runner.invoke(app, ["list-generators"])
    assert result.exit_code == 0
    assert "ArtificialGrammar" in result.stdout


def test_validate_example_config_succeeds():
    result = runner.invoke(app, ["validate", "examples/configs/random_ag.yaml"])
    assert result.exit_code == 0
    assert "Valid config" in result.stdout


def test_validate_requires_storage_path(tmp_path):
    config_path = tmp_path / "missing_storage.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            symseq:
              seed: 42
              generator:
                type: NBack
                params: {n: 2, alphabet_size: 6, seq_length: 12}
              trial_set:
                n_trials: 2
            """
        )
    )

    result = runner.invoke(app, ["validate", str(config_path)])
    assert result.exit_code != 0
    assert "symseq.storage.path" in result.output


def test_inspect_prints_preview_without_writing(tmp_path):
    output_dir = tmp_path / "out"
    config_path = tmp_path / "random_ag.yaml"
    _write_yaml(config_path, _random_ag_config(output_dir))

    result = runner.invoke(app, ["inspect", str(config_path), "--head", "2"])
    assert result.exit_code == 0
    assert not output_dir.exists()

    rows = [json.loads(line) for line in result.stdout.strip().splitlines()]
    assert len(rows) == 2
    assert {"index", "symbols", "states", "targets", "meta"} <= rows[0].keys()


def test_generate_writes_symbolic_dataset(tmp_path):
    output_dir = tmp_path / "out"
    config_path = tmp_path / "random_ag.yaml"
    _write_yaml(config_path, _random_ag_config(output_dir))

    result = runner.invoke(app, ["generate", str(config_path)])
    assert result.exit_code == 0

    assert (output_dir / "config.yaml").exists()
    assert (output_dir / "manifest.yaml").exists()
    assert (output_dir / "trials.jsonl").exists()
    assert (output_dir / "splits" / "train.jsonl").exists()
    assert (output_dir / "splits" / "test.jsonl").exists()

    trial_rows = [json.loads(line) for line in (output_dir / "trials.jsonl").read_text().splitlines()]
    train_rows = (output_dir / "splits" / "train.jsonl").read_text().splitlines()
    test_rows = (output_dir / "splits" / "test.jsonl").read_text().splitlines()

    assert len(trial_rows) == 6
    assert len(train_rows) == 4
    assert len(test_rows) == 2
    assert {"index", "symbols", "states", "targets", "meta"} <= trial_rows[0].keys()
