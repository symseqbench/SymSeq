"""
test_trial_set.py

Tests for the TrialSet class and its TrialSource Protocol conformance.
"""

import pytest

from symseq.trial import Trial, Target
from symseq.trial_set import TrialSet
from symseq.trial_source import TrialSource


def _mk_trial(symbols: list[str]) -> Trial:
    return Trial(symbols=symbols, states=None, targets={}, meta={})


def _ts(n: int = 10, splits=None, alphabet=None) -> TrialSet:
    trials = [_mk_trial([f"t{i}_a", f"t{i}_b"]) for i in range(n)]
    return TrialSet(
        trials=trials,
        splits=splits or {},
        meta={"alphabet": alphabet or ["a", "b"]},
    )


class TestConstruction:
    def test_basic(self):
        ts = _ts(n=5)
        assert len(ts) == 5
        assert ts.alphabet == ["a", "b"]
        assert ts.splits == {}

    def test_split_index_validation(self):
        trials = [_mk_trial(["x"]) for _ in range(3)]
        with pytest.raises(IndexError, match="out-of-range"):
            TrialSet(trials=trials, splits={"train": [0, 1, 5]})

    def test_negative_index_rejected(self):
        trials = [_mk_trial(["x"]) for _ in range(3)]
        with pytest.raises(IndexError, match="out-of-range"):
            TrialSet(trials=trials, splits={"train": [-1]})

    def test_default_draw_split_is_train_when_present(self):
        ts = _ts(n=10, splits={"train": [0, 1, 2], "test": [3, 4]})
        assert ts._draw_split == "train"

    def test_default_draw_split_is_none_without_train(self):
        ts = _ts(n=10, splits={"foo": [0, 1, 2]})
        assert ts._draw_split is None


class TestCollectionInterface:
    def test_iter(self):
        ts = _ts(n=4)
        symbols_seen = [t.symbols for t in ts]
        assert len(symbols_seen) == 4
        assert symbols_seen[0] == ["t0_a", "t0_b"]

    def test_getitem(self):
        ts = _ts(n=4)
        assert ts[2].symbols == ["t2_a", "t2_b"]

    def test_split_lookup(self):
        ts = _ts(n=10, splits={"train": [0, 1, 2], "test": [7, 8, 9]})
        train = ts.split("train")
        test = ts.split("test")
        assert [t.symbols[0] for t in train] == ["t0_a", "t1_a", "t2_a"]
        assert [t.symbols[0] for t in test] == ["t7_a", "t8_a", "t9_a"]

    def test_split_unknown_name(self):
        ts = _ts(n=5, splits={"train": [0, 1]})
        with pytest.raises(KeyError, match="unknown split 'val'"):
            ts.split("val")


class TestDrawAndProtocol:
    def test_implements_trial_source(self):
        ts = _ts(n=5, splits={"train": [0, 1, 2]})
        assert isinstance(ts, TrialSource)

    def test_draw_trial_cycles_default_split(self):
        ts = _ts(n=5, splits={"train": [0, 1, 2]})
        drawn = [ts.draw_trial() for _ in range(7)]
        # cycles deterministically through indices 0,1,2,0,1,2,0
        assert [t.symbols[0] for t in drawn] == [
            "t0_a",
            "t1_a",
            "t2_a",
            "t0_a",
            "t1_a",
            "t2_a",
            "t0_a",
        ]

    def test_draw_batch(self):
        ts = _ts(n=5, splits={"train": [0, 1, 2]})
        batch = ts.draw_batch(4)
        assert len(batch) == 4
        assert [t.symbols[0] for t in batch] == ["t0_a", "t1_a", "t2_a", "t0_a"]

    def test_draw_falls_back_to_all_trials_when_no_train(self):
        ts = _ts(n=3, splits={})
        drawn = [ts.draw_trial() for _ in range(5)]
        assert [t.symbols[0] for t in drawn] == ["t0_a", "t1_a", "t2_a", "t0_a", "t1_a"]

    def test_set_draw_split_switches_source(self):
        ts = _ts(n=10, splits={"train": [0, 1, 2], "test": [7, 8, 9]})
        ts.set_draw_split("test")
        drawn = [ts.draw_trial() for _ in range(4)]
        assert [t.symbols[0] for t in drawn] == ["t7_a", "t8_a", "t9_a", "t7_a"]

    def test_set_draw_split_resets_cursor(self):
        ts = _ts(n=10, splits={"train": [0, 1, 2], "test": [7, 8, 9]})
        ts.draw_trial()  # advance cursor on train
        ts.draw_trial()
        ts.set_draw_split("test")  # cursor reset
        drawn = [ts.draw_trial() for _ in range(2)]
        assert [t.symbols[0] for t in drawn] == ["t7_a", "t8_a"]

    def test_set_draw_split_to_none_iterates_all(self):
        ts = _ts(n=3, splits={"train": [0, 1]})
        ts.set_draw_split(None)
        drawn = [ts.draw_trial() for _ in range(4)]
        assert [t.symbols[0] for t in drawn] == ["t0_a", "t1_a", "t2_a", "t0_a"]

    def test_set_draw_split_unknown_raises(self):
        ts = _ts(n=3, splits={"train": [0, 1]})
        with pytest.raises(KeyError):
            ts.set_draw_split("nope")

    def test_reset_cursor(self):
        ts = _ts(n=5, splits={"train": [0, 1, 2]})
        ts.draw_trial()
        ts.draw_trial()
        ts.reset_cursor()
        first_again = ts.draw_trial()
        assert first_again.symbols[0] == "t0_a"

    def test_empty_split_raises(self):
        trials = [_mk_trial(["x"]) for _ in range(2)]
        ts = TrialSet(trials=trials, splits={"train": []})
        with pytest.raises(IndexError, match="is empty"):
            ts.draw_trial()

    def test_empty_trial_set_raises(self):
        ts = TrialSet(trials=[], splits={})
        with pytest.raises(IndexError, match="empty"):
            ts.draw_trial()


class TestAlphabet:
    def test_alphabet_from_meta(self):
        ts = _ts(n=3, alphabet=["a", "b", "c"])
        assert ts.alphabet == ["a", "b", "c"]

    def test_alphabet_default_empty_if_missing(self):
        trials = [_mk_trial(["x"]) for _ in range(2)]
        ts = TrialSet(trials=trials, meta={})  # no "alphabet" key
        assert ts.alphabet == []
