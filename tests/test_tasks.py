"""
test_tasks.py

Tests for the Task ABC and the built-in task classes (NStepMemory,
NStepPrediction, NGramChunk).
"""

import pytest

from symseq.tasks import NGramChunk, NStepMemory, NStepPrediction, Task
from symseq.tasks.base import Task as TaskBase
from symseq.tasks.registry import build, registered_names
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
            name = "incomplete"

        with pytest.raises(TypeError):
            Incomplete()

    def test_class_alias_in_public_api(self):
        assert Task is TaskBase


class TestRegistry:
    def test_registered_names(self):
        names = registered_names()
        assert "NStepMemory" in names
        assert "NStepPrediction" in names
        assert "NGramChunk" in names

    def test_build_n_step_memory(self):
        t = build("NStepMemory", n=2)
        assert isinstance(t, NStepMemory)
        assert t.n == 2

    def test_build_unknown_task_raises(self):
        with pytest.raises(KeyError, match="Unknown task"):
            build("DoesNotExist")


# --------------------- NStepMemory ---------------------


class TestNStepMemory:
    def test_basic_1_step(self):
        task = NStepMemory(n=1)
        trial = _trial(["A", "B", "C", "D"])
        tgt = task(trial)
        assert isinstance(tgt, Target)
        assert tgt.kind == "per_token"
        # at position 0: masked; at i>=1: target = symbols[i-1]
        assert tgt.values == [None, "A", "B", "C"]
        assert tgt.mask == [False, True, True, True]

    def test_basic_3_step(self):
        task = NStepMemory(n=3)
        trial = _trial(["A", "B", "C", "D", "E"])
        tgt = task(trial)
        assert tgt.values == [None, None, None, "A", "B"]
        assert tgt.mask == [False, False, False, True, True]

    def test_name_includes_n(self):
        assert NStepMemory(n=3).name == "3_step_memory"

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
        trial.targets[task.name] = task(trial)
        assert "2_step_memory" in trial.targets


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

    def test_name_includes_n(self):
        assert NStepPrediction(n=4).name == "4_step_prediction"

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
        assert tgt.kind == "per_token"
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

    def test_name_includes_n(self):
        assert NGramChunk(n=2).name == "2_gram_chunk"

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
        # generator-intrinsic + task-derived targets coexist
        memory = NStepMemory(n=3)
        trial.targets[memory.name] = memory(trial)
        assert "nback_match" in trial.targets       # intrinsic
        assert "3_step_memory" in trial.targets     # derived
        assert len(trial.targets["3_step_memory"].values) == 10

    def test_apply_prediction_and_chunk_to_dyck_trial(self):
        import numpy as np
        from symseq.generators.dyck import DyckGenerator

        gen = DyckGenerator(k=2, mode="uniform", target_pairs=4, rng=np.random.default_rng(42))
        trial = gen.generate_trial()
        pred = NStepPrediction(n=1)
        chunk = NGramChunk(n=2)
        trial.targets[pred.name] = pred(trial)
        trial.targets[chunk.name] = chunk(trial)
        L = len(trial.symbols)
        assert len(trial.targets["1_step_prediction"].values) == L
        assert len(trial.targets["2_gram_chunk"].values) == L
