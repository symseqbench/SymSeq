"""
test_config.py

Tests for symseq.config.load_trial_set: schema validation, generator
construction via registry, split resolution, and meta population.
"""

import textwrap

import pytest

from symseq.config import load_trial_set, _resolve_splits
from symseq.generators.nback import NBack
from symseq.trial import Trial
from symseq.trial_set import TrialSet
from symseq.trial_source import TrialSource


# --------------------- helpers ---------------------


def _nback_cfg(**overrides) -> dict:
    cfg = {
        "symseq": {
            "seed": 42,
            "generator": {
                "type": "NBack",
                "params": {"n": 2, "alphabet_size": 6, "seq_length": 15},
            },
            "trial_set": {
                "n_trials": 10,
                "splits": {"train": 7, "test": 3},
            },
        }
    }
    for k, v in overrides.items():
        cfg["symseq"][k] = v
    return cfg


# --------------------- _resolve_splits ---------------------


class TestResolveSplits:
    def test_integer_splits(self):
        out = _resolve_splits(10, {"train": 7, "test": 3})
        assert out["train"] == list(range(0, 7))
        assert out["test"] == list(range(7, 10))

    def test_fractional_splits(self):
        out = _resolve_splits(100, {"train": 0.8, "test": 0.2})
        assert len(out["train"]) == 80
        assert len(out["test"]) == 20
        assert out["train"] == list(range(0, 80))
        assert out["test"] == list(range(80, 100))

    def test_mixed_int_and_fraction(self):
        out = _resolve_splits(100, {"train": 0.5, "test": 30})
        assert len(out["train"]) == 50
        assert len(out["test"]) == 30

    def test_split_overflow_raises(self):
        with pytest.raises(ValueError, match="exceed"):
            _resolve_splits(10, {"train": 7, "test": 5})

    def test_negative_int_rejected(self):
        with pytest.raises(ValueError, match="int>=0"):
            _resolve_splits(10, {"train": -3})

    def test_fraction_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="int>=0 or float"):
            _resolve_splits(10, {"train": 1.5})

    def test_empty_splits_dict(self):
        assert _resolve_splits(10, {}) == {}


# --------------------- load_trial_set from dict ---------------------


class TestLoadFromDict:
    def test_basic_nback(self):
        ts = load_trial_set(_nback_cfg())
        assert isinstance(ts, TrialSet)
        assert len(ts) == 10
        assert ts.splits["train"] == list(range(0, 7))
        assert ts.splits["test"] == list(range(7, 10))

    def test_meta_populated(self):
        ts = load_trial_set(_nback_cfg())
        assert ts.meta["seed"] == 42
        assert isinstance(ts.meta["generator"], NBack)
        assert isinstance(ts.meta["alphabet"], list)
        assert "config" in ts.meta

    def test_trials_have_intrinsic_targets(self):
        ts = load_trial_set(_nback_cfg())
        for trial in ts.trials:
            assert isinstance(trial, Trial)
            assert "nback_match" in trial.targets
            assert "nback_role" in trial.targets

    def test_trial_set_satisfies_trial_source(self):
        ts = load_trial_set(_nback_cfg())
        assert isinstance(ts, TrialSource)
        t = ts.draw_trial()
        assert isinstance(t, Trial)

    def test_reproducible_with_seed(self):
        ts1 = load_trial_set(_nback_cfg())
        ts2 = load_trial_set(_nback_cfg())
        assert [t.symbols for t in ts1.trials] == [t.symbols for t in ts2.trials]

    def test_different_seed_different_trials(self):
        ts1 = load_trial_set(_nback_cfg(seed=1))
        ts2 = load_trial_set(_nback_cfg(seed=2))
        assert [t.symbols for t in ts1.trials] != [t.symbols for t in ts2.trials]

    def test_no_splits_section_yields_empty_splits(self):
        cfg = _nback_cfg()
        cfg["symseq"]["trial_set"].pop("splits")
        ts = load_trial_set(cfg)
        assert ts.splits == {}
        assert len(ts) == 10

    def test_gen_params_forwarded(self):
        cfg = _nback_cfg()
        cfg["symseq"]["trial_set"]["gen_params"] = {"seq_length": 25}
        ts = load_trial_set(cfg)
        assert all(len(t.symbols) == 25 for t in ts.trials)


class TestLoadFromDictArtificialGrammarPreset:
    def test_preset_construction(self):
        cfg = {
            "symseq": {
                "seed": 42,
                "generator": {"type": "ArtificialGrammar", "preset": "Elman"},
                "trial_set": {
                    "n_trials": 5,
                    "gen_params": {"length_range": [3, 20]},
                },
            }
        }
        ts = load_trial_set(cfg)
        assert len(ts) == 5
        for t in ts.trials:
            assert t.symbols
            assert t.states is not None
            assert t.targets["grammaticality"].values is True


# --------------------- load_trial_set from file ---------------------


class TestLoadFromYAMLFile:
    def test_load_yaml(self, tmp_path):
        yaml_content = textwrap.dedent(
            """
            symseq:
              seed: 42
              generator:
                type: NBack
                params:
                  n: 2
                  alphabet_size: 6
                  seq_length: 12
              trial_set:
                n_trials: 8
                splits:
                  train: 6
                  test: 2
            """
        )
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml_content)
        ts = load_trial_set(path)
        assert len(ts) == 8
        assert len(ts.split("train")) == 6
        assert len(ts.split("test")) == 2

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_trial_set(tmp_path / "nope.yaml")

    def test_unsupported_extension(self, tmp_path):
        path = tmp_path / "cfg.json"
        path.write_text("{}")
        with pytest.raises(ValueError, match="unsupported config extension"):
            load_trial_set(path)


# --------------------- validation errors ---------------------


class TestValidation:
    def test_missing_symseq_section(self):
        with pytest.raises(ValueError, match="missing top-level 'symseq'"):
            load_trial_set({})

    def test_missing_generator(self):
        with pytest.raises(ValueError, match="missing 'generator'"):
            load_trial_set({"symseq": {"trial_set": {"n_trials": 1}}})

    def test_missing_generator_type(self):
        with pytest.raises(ValueError, match="missing 'type'"):
            load_trial_set(
                {"symseq": {"generator": {}, "trial_set": {"n_trials": 1}}}
            )

    def test_missing_trial_set(self):
        with pytest.raises(ValueError, match="missing 'trial_set'"):
            load_trial_set({"symseq": {"generator": {"type": "NBack"}}})

    def test_missing_n_trials(self):
        with pytest.raises(ValueError, match="missing 'n_trials'"):
            load_trial_set(
                {"symseq": {"generator": {"type": "NBack"}, "trial_set": {}}}
            )

    def test_unknown_generator_type(self):
        cfg = _nback_cfg()
        cfg["symseq"]["generator"]["type"] = "DoesNotExist"
        with pytest.raises(KeyError, match="Unknown generator"):
            load_trial_set(cfg)
