"""
test_tasks.py

Tests for the Task ABC and the built-in task classes (NStepMemory,
NStepPrediction, NGramChunk).
"""

import pickle
from types import SimpleNamespace

import pytest

from symseq.tasks import (
    ConfiguredTrialSource,
    NAXIsTarget,
    NGramChunk,
    NStepMemory,
    NStepPrediction,
    Task,
    build_tasks,
    coerce_task_entries,
    materialize_targets,
)
from symseq.tasks.base import Task as TaskBase
from symseq.tasks.registry import build, registered_types
from symseq.trial import Target, Trial


def _trial(symbols: list[str]) -> Trial:
    return Trial(symbols=symbols, states=None, targets={}, meta={})


# --------------------- ABC + registry ---------------------


class TestTaskBase:
    def test_task_abc_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            TaskBase()

    def test_subclass_must_implement_call(self):
        class Incomplete(TaskBase):
            granularity = "per_token"

        with pytest.raises(TypeError):
            Incomplete()

    def test_class_alias_in_public_api(self):
        assert Task is TaskBase


class TestRegistry:
    def test_registered_types(self):
        types = registered_types()
        assert "NStepMemory" in types
        assert "NStepPrediction" in types
        assert "NGramChunk" in types
        assert "Grammaticality" in types

    def test_build_n_step_memory(self):
        t = build("NStepMemory", n=2)
        assert isinstance(t, NStepMemory)
        assert t.n == 2

    def test_build_unknown_task_raises(self):
        with pytest.raises(KeyError, match="Unknown task"):
            build("DoesNotExist")

    def test_generated_intrinsic_task_types_preserve_public_identity(self):
        task = build("NAXIsTarget")
        assert isinstance(task, NAXIsTarget)
        assert type(task).__name__ == "NAXIsTarget"
        assert task.intrinsic_id == "nax_is_target"
        assert task.granularity == "per_trial"
        assert isinstance(pickle.loads(pickle.dumps(task)), NAXIsTarget)


class TestConfiguredTasks:
    def test_builds_mapping_keyed_by_configured_id(self):
        tasks = build_tasks(
            [{"id": "prediction", "type": "NStepPrediction", "params": {"n": 2}}]
        )
        assert list(tasks) == ["prediction"]
        assert isinstance(tasks["prediction"], NStepPrediction)
        assert tasks["prediction"].n == 2

    def test_accepts_typed_config_entries(self):
        tasks = build_tasks(
            [SimpleNamespace(id="memory", type="NStepMemory", params={"n": 1})]
        )
        assert isinstance(tasks["memory"], NStepMemory)

    def test_coercion_returns_fresh_normalized_entries(self):
        params = {"n": 1}
        entries = coerce_task_entries(
            [SimpleNamespace(id="memory", type="NStepMemory", params=params)]
        )
        assert entries == [("memory", "NStepMemory", {"n": 1})]
        assert entries[0][2] is not params

    def test_none_params_are_normalized_to_empty_mapping(self):
        entries = coerce_task_entries([{"id": "match", "type": "NBackMatch", "params": None}])
        assert entries == [("match", "NBackMatch", {})]

    def test_requires_id_when_built_directly(self):
        with pytest.raises(ValueError, match=r"tasks\[0\]\.id.*non-empty string"):
            build_tasks([{"type": "NStepPrediction", "params": {"n": 1}}])

    @pytest.mark.parametrize("params", [[1], []])
    def test_requires_mapping_params_when_built_directly(self, params):
        with pytest.raises(ValueError, match="params must be a mapping"):
            build_tasks(
                [{"id": "prediction", "type": "NStepPrediction", "params": params}]
            )

    def test_mapping_entries_reject_unknown_keys(self):
        with pytest.raises(ValueError, match=r"unknown keys.*name"):
            build_tasks([{"id": "prediction", "type": "NStepPrediction", "name": "old"}])

    def test_entries_must_be_a_list(self):
        with pytest.raises(ValueError, match="must be a list"):
            build_tasks(({"id": "prediction", "type": "NStepPrediction"},))

    def test_duplicate_ids_are_rejected_when_built_directly(self):
        with pytest.raises(ValueError, match="must be unique"):
            build_tasks(
                [
                    {"id": "prediction", "type": "NStepPrediction", "params": {"n": 1}},
                    {"id": "prediction", "type": "NStepPrediction", "params": {"n": 2}},
                ]
            )

    def test_empty_tasks_replace_existing_public_targets(self):
        trial = Trial(
            symbols=["A"],
            targets={
                "old": Target(values=True, mask=None, granularity="per_trial")
            },
        )
        result = materialize_targets(trial, {})
        assert result is None
        assert trial.targets == {}

    def test_intrinsic_task_compatibility_is_checked_on_wrapper_construction(self):
        from symseq.generators.nback import NBack

        source = NBack(n=2, seq_length=8, alphabet_size=5, seed=1)
        tasks = build_tasks([{"id": "valid", "type": "Grammaticality"}])
        with pytest.raises(ValueError, match=r"requires intrinsic target.*grammaticality"):
            ConfiguredTrialSource(source, tasks)

    def test_generic_tasks_do_not_require_intrinsic_capabilities(self):
        from symseq.generators.nback import NBack

        source = NBack(n=2, seq_length=8, alphabet_size=5, seed=1)
        tasks = build_tasks([{"id": "prediction", "type": "NStepPrediction", "params": {"n": 1}}])
        assert ConfiguredTrialSource(source, tasks).draw_trial().targets["prediction"]

    def test_intrinsic_task_copies_mutable_target_fields(self):
        trial = Trial(
            symbols=["A", "B"],
            intrinsic_targets={
                "nback_match": Target(
                    values=[None, 1], mask=[False, True], granularity="per_token"
                )
            },
        )
        task = build_tasks([{"id": "match", "type": "NBackMatch"}])["match"]
        target = task(trial)
        assert target.values is not trial.intrinsic_targets["nback_match"].values
        assert target.mask is not trial.intrinsic_targets["nback_match"].mask

    def test_intrinsic_task_rejects_runtime_granularity_mismatch(self):
        trial = Trial(
            symbols=["A"],
            intrinsic_targets={
                "nback_match": Target(
                    values=True, mask=None, granularity="per_trial"
                )
            },
        )
        task = build("NBackMatch")
        with pytest.raises(ValueError, match=r"expects.*per_token.*got.*per_trial"):
            task(trial)


# --------------------- NStepMemory ---------------------


class TestNStepMemory:
    def test_basic_1_step(self):
        task = NStepMemory(n=1)
        trial = _trial(["A", "B", "C", "D"])
        tgt = task(trial)
        assert isinstance(tgt, Target)
        assert tgt.granularity == "per_token"
        # at position 0: masked; at i>=1: target = symbols[i-1]
        assert tgt.values == [None, "A", "B", "C"]
        assert tgt.mask == [False, True, True, True]

    def test_basic_3_step(self):
        task = NStepMemory(n=3)
        trial = _trial(["A", "B", "C", "D", "E"])
        tgt = task(trial)
        assert tgt.values == [None, None, None, "A", "B"]
        assert tgt.mask == [False, False, False, True, True]

    def test_invalid_n_zero_rejected(self):
        with pytest.raises(ValueError, match="positive int"):
            NStepMemory(n=0)

    def test_invalid_n_negative_rejected(self):
        with pytest.raises(ValueError, match="positive int"):
            NStepMemory(n=-1)

    def test_invalid_n_non_int_rejected(self):
        with pytest.raises(ValueError, match="positive int"):
            NStepMemory(n=2.0)

    def test_n_larger_than_sequence_masks_all(self):
        task = NStepMemory(n=5)
        trial = _trial(["A", "B", "C"])
        tgt = task(trial)
        assert tgt.values == [None, None, None]
        assert tgt.mask == [False, False, False]

    def test_n_equal_to_sequence_length_masks_all(self):
        task = NStepMemory(n=3)
        trial = _trial(["A", "B", "C"])
        tgt = task(trial)
        assert tgt.mask == [False, False, False]

    def test_attach_to_trial_targets(self):
        task = NStepMemory(n=2)
        trial = _trial(["A", "B", "C", "D"])
        trial.targets["memory"] = task(trial)
        assert "memory" in trial.targets


# --------------------- NStepPrediction ---------------------


class TestNStepPrediction:
    def test_basic_1_step(self):
        task = NStepPrediction(n=1)
        trial = _trial(["A", "B", "C", "D"])
        tgt = task(trial)
        # at position i: target = symbols[i+1]; last masked
        assert tgt.values == ["B", "C", "D", None]
        assert tgt.mask == [True, True, True, False]

    def test_basic_2_step(self):
        task = NStepPrediction(n=2)
        trial = _trial(["A", "B", "C", "D", "E"])
        tgt = task(trial)
        assert tgt.values == ["C", "D", "E", None, None]
        assert tgt.mask == [True, True, True, False, False]

    def test_invalid_n_rejected(self):
        with pytest.raises(ValueError, match="positive int"):
            NStepPrediction(n=0)

    def test_n_larger_than_sequence_masks_all(self):
        task = NStepPrediction(n=10)
        trial = _trial(["A", "B"])
        tgt = task(trial)
        assert tgt.values == [None, None]
        assert tgt.mask == [False, False]


# --------------------- NGramChunk ---------------------


class TestNGramChunk:
    def test_unigram_is_identity(self):
        task = NGramChunk(n=1)
        trial = _trial(["A", "B", "C"])
        tgt = task(trial)
        assert tgt.granularity == "per_token"
        assert tgt.values == [("A",), ("B",), ("C",)]
        assert tgt.mask == [True, True, True]

    def test_bigram_masks_first_position(self):
        task = NGramChunk(n=2)
        trial = _trial(["A", "B", "C", "D"])
        tgt = task(trial)
        assert tgt.values == [None, ("A", "B"), ("B", "C"), ("C", "D")]
        assert tgt.mask == [False, True, True, True]

    def test_trigram_masks_first_two(self):
        task = NGramChunk(n=3)
        trial = _trial(["A", "B", "C", "D", "E"])
        tgt = task(trial)
        assert tgt.values == [None, None, ("A", "B", "C"), ("B", "C", "D"), ("C", "D", "E")]
        assert tgt.mask == [False, False, True, True, True]

    def test_invalid_n_rejected(self):
        with pytest.raises(ValueError, match="positive int"):
            NGramChunk(n=0)

    def test_n_larger_than_sequence_masks_all(self):
        task = NGramChunk(n=5)
        trial = _trial(["A", "B"])
        tgt = task(trial)
        assert tgt.values == [None, None]
        assert tgt.mask == [False, False]


# --------------------- integration: apply tasks to generated trials ---------------------


class TestIntegrationWithGenerators:
    def test_apply_memory_to_nback_trial(self):
        from symseq.generators.nback import NBack

        gen = NBack(n=2, seq_length=10, alphabet_size=6, seed=42)
        trial = gen.generate_trial()
        tasks = build_tasks(
            [
                {"id": "match", "type": "NBackMatch"},
                {"id": "memory", "type": "NStepMemory", "params": {"n": 3}},
            ]
        )
        materialize_targets(trial, tasks)
        assert set(trial.targets) == {"match", "memory"}
        assert len(trial.targets["memory"].values) == 10

    def test_apply_prediction_and_chunk_to_dyck_trial(self):
        import numpy as np

        from symseq.generators.dyck import DyckGenerator

        gen = DyckGenerator(k=2, mode="uniform", target_pairs=4, rng=np.random.default_rng(42))
        trial = gen.generate_trial()
        tasks = build_tasks(
            [
                {"id": "prediction", "type": "NStepPrediction", "params": {"n": 1}},
                {"id": "chunk", "type": "NGramChunk", "params": {"n": 2}},
            ]
        )
        materialize_targets(trial, tasks)
        L = len(trial.symbols)
        assert len(trial.targets["prediction"].values) == L
        assert len(trial.targets["chunk"].values) == L

    def test_intrinsic_task_is_published_only_when_configured(self):
        from symseq.generators.ag import ArtificialGrammar

        gen = ArtificialGrammar.from_preset("Elman", seed=42)
        trial = gen.generate_trial(length_range=[3, 10])
        assert trial.targets == {}
        assert "grammaticality" in trial.intrinsic_targets

        tasks = build_tasks([{"id": "is_valid", "type": "Grammaticality"}])
        materialize_targets(trial, tasks)
        assert set(trial.targets) == {"is_valid"}
        assert trial.targets["is_valid"].values is True
