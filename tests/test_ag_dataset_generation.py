"""Regression tests for ArtificialGrammar batch generation."""

from __future__ import annotations

from math import floor

import numpy as np
import pytest

from symseq.generators.ag import ArtificialGrammar
from symseq.utils.strtools import string_as_symbols


def _linear_grammar(seed: int = 7) -> ArtificialGrammar:
    return ArtificialGrammar(
        label="linear",
        states=["A", "B"],
        alphabet=["A", "B"],
        transitions=[("A", "B", 1.0)],
        start_states=["A"],
        terminal_states=["B"],
        seed=seed,
    )


def _branching_grammar(seed: int = 7) -> ArtificialGrammar:
    return ArtificialGrammar(
        label="branching",
        states=["A", "B", "T"],
        alphabet=["A", "B", "T"],
        transitions=[
            ("A", "A", 0.25),
            ("A", "B", 0.25),
            ("A", "T", 0.5),
            ("B", "A", 0.25),
            ("B", "B", 0.25),
            ("B", "T", 0.5),
        ],
        start_states=["A", "B"],
        terminal_states=["T"],
        seed=seed,
    )


def _ambiguous_grammar(seed: int = 7) -> ArtificialGrammar:
    return ArtificialGrammar(
        label="ambiguous",
        states=["S", "B(1)", "B(2)", "C", "D"],
        alphabet=["S", "B", "C", "D"],
        transitions=[
            ("S", "B(1)", 0.5),
            ("S", "B(2)", 0.5),
            ("B(1)", "C", 1.0),
            ("B(2)", "D", 1.0),
        ],
        start_states=["S"],
        terminal_states=["C", "D"],
        seed=seed,
    )


@pytest.mark.parametrize(
    ("n_samples", "fraction"),
    [(0, 0.0), (1, 0.5), (3, 0.5), (5, 0.3), (6, 1.0)],
)
def test_batch_size_and_floor_rounded_labels_are_exact(n_samples, fraction):
    grammar = _linear_grammar()

    strings, labels = grammar.generate_string_set(
        n_samples,
        length_range=(2, 2),
        frac_violations=fraction,
    )

    assert len(strings) == n_samples
    assert len(labels) == n_samples
    assert labels.count(False) == floor(fraction * n_samples)


def test_labels_match_symbol_level_grammar_oracle():
    grammar = _branching_grammar()

    strings, labels = grammar.generate_string_set(
        40,
        length_range=(2, 8),
        frac_violations=0.4,
    )

    assert all(grammar.is_grammatical(string) is label for string, label in zip(strings, labels, strict=True))


def test_state_violations_are_also_nongrammatical_as_symbols():
    grammar = _ambiguous_grammar(seed=2)

    strings, labels = grammar.generate_string_set(
        20,
        length_range=(3, 3),
        frac_violations=1.0,
        as_states=True,
    )

    assert labels == [False] * 20
    assert all(not grammar.is_grammatical(states) for states in strings)
    assert all(not grammar.is_grammatical(string_as_symbols(states)) for states in strings)


def test_add_deviant_does_not_mutate_input_and_accepts_one_symbol():
    grammar = ArtificialGrammar(
        label="one-symbol-string",
        states=["A", "B"],
        alphabet=["A", "B"],
        transitions=[("B", "B", 1.0)],
        start_states=["A"],
        terminal_states=["A"],
        seed=4,
    )
    original = ["A"]

    deviant = grammar.add_deviant(original)

    assert original == ["A"]
    assert deviant == ["B"]
    assert not grammar.is_grammatical(deviant)


def test_generate_nongrammatical_string_forwards_corruption_attempt_limit(monkeypatch):
    grammar = _linear_grammar()
    observed = {}
    original = grammar.add_deviant

    def record_max_iter(string, **kwargs):
        observed["max_iter"] = kwargs["max_iter"]
        return original(string, **kwargs)

    monkeypatch.setattr(grammar, "add_deviant", record_max_iter)

    grammar.generate_nongrammatical_string(length_range=(2, 2), max_iter=17)

    assert observed["max_iter"] == 17


def test_multiple_deviants_condition_generation_on_corruptible_lengths():
    grammar = _branching_grammar(seed=12)

    strings, labels = grammar.generate_string_set(
        12,
        length_range=(1, 6),
        frac_violations=1.0,
        n_deviants=3,
    )

    assert all(len(string) >= 3 for string in strings)
    assert all(label is False for label in labels)
    assert all(not grammar.is_grammatical(string) for string in strings)


def test_impossible_deviant_count_is_rejected_before_generation():
    grammar = _linear_grammar()

    with pytest.raises(ValueError, match="maximum generated string length"):
        grammar.generate_string_set(
            1,
            length_range=(1, 2),
            frac_violations=1.0,
            n_deviants=3,
        )


def test_without_replacement_returns_unique_requested_outputs():
    grammar = _branching_grammar(seed=15)

    strings, labels = grammar.generate_string_set(
        25,
        length_range=(2, 8),
        frac_violations=0.2,
        replace=False,
        max_candidates=1000,
    )

    assert len(strings) == 25
    assert len({tuple(string) for string in strings}) == 25
    assert labels.count(False) == 5


def test_strict_unique_sampling_raises_at_candidate_limit():
    grammar = _linear_grammar()

    with pytest.raises(RuntimeError, match="1/3 grammatical"):
        grammar.generate_string_set(
            3,
            length_range=(2, 2),
            replace=False,
            max_candidates=3,
        )


def test_nonstrict_partial_batch_preserves_returned_fraction_and_uniqueness():
    grammar = _linear_grammar()

    with pytest.warns(UserWarning, match="Generated 2/6"):
        strings, labels = grammar.generate_string_set(
            6,
            length_range=(2, 2),
            frac_violations=0.5,
            replace=False,
            strict=False,
            max_candidates=60,
        )

    assert len(strings) == 2
    assert len({tuple(string) for string in strings}) == 2
    assert labels.count(False) == floor(0.5 * len(strings))
    assert all(grammar.is_grammatical(string) is label for string, label in zip(strings, labels, strict=True))


def test_seeded_output_is_independent_of_worker_count():
    expected = _branching_grammar(seed=2026).generate_string_set(
        30,
        length_range=(2, 8),
        frac_violations=0.3,
        n_proc=1,
    )

    for workers in (2, 4):
        actual = _branching_grammar(seed=2026).generate_string_set(
            30,
            length_range=(2, 8),
            frac_violations=0.3,
            n_proc=workers,
        )
        assert actual == expected


def test_trial_batches_share_fraction_uniqueness_and_oracle_semantics():
    grammar = _branching_grammar(seed=31)

    trials = grammar.generate_trials(
        20,
        length_range=(2, 8),
        frac_violations=0.25,
        replace=False,
        max_candidates=1000,
        n_proc=2,
    )
    labels = [trial.intrinsic_targets["grammaticality"].values for trial in trials]

    assert len(trials) == 20
    assert len({tuple(trial.symbols) for trial in trials}) == 20
    assert labels.count(False) == 5
    assert all(grammar.is_grammatical(trial.symbols) is label for trial, label in zip(trials, labels, strict=True))
    assert all(trial.meta["length"] == len(trial.symbols) for trial in trials)


def test_draw_batch_forwards_ag_batch_controls():
    grammar = _linear_grammar()

    trials = grammar.draw_batch(5, length_range=(2, 2), frac_violations=0.4)
    labels = [trial.intrinsic_targets["grammaticality"].values for trial in trials]

    assert labels.count(False) == 2


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"n_samples": -1}, ValueError),
        ({"n_samples": 1.5}, TypeError),
        ({"n_samples": True}, TypeError),
        ({"n_samples": 1, "frac_violations": -0.1}, ValueError),
        ({"n_samples": 1, "frac_violations": np.nan}, ValueError),
        ({"n_samples": 1, "frac_violations": "half"}, TypeError),
        ({"n_samples": 1, "replace": 1}, TypeError),
        ({"n_samples": 1, "strict": 1}, TypeError),
        ({"n_samples": 1, "n_deviants": 0}, ValueError),
        ({"n_samples": 1, "max_iter": 0}, ValueError),
        ({"n_samples": 1, "n_proc": 0}, ValueError),
        ({"n_samples": 1, "n_proc": 1.5}, TypeError),
        ({"n_samples": 2, "max_candidates": 1}, ValueError),
        ({"n_samples": 1, "length_range": (3, 2)}, ValueError),
        ({"n_samples": 1, "length_range": (1.0, 2)}, ValueError),
    ],
)
def test_batch_argument_validation(kwargs, error):
    grammar = _linear_grammar()

    with pytest.raises(error):
        grammar.generate_string_set(**kwargs)


def test_removed_legacy_batch_keywords_are_not_accepted():
    grammar = _linear_grammar()

    with pytest.raises(TypeError, match="nongramm_fraction"):
        grammar.generate_string_set(1, nongramm_fraction=0.5)
    with pytest.raises(TypeError, match="allow_repetitions"):
        grammar.generate_string_set(1, allow_repetitions=False)
    with pytest.raises(TypeError, match="oversample_factor"):
        grammar.generate_string_set(1, oversample_factor=2.0)
