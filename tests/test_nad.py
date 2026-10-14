"""Behavioral tests for all non-adjacent dependency generators."""

import logging
from collections import Counter

import numpy as np
import pytest

from symseq.generators import (
    CrossedNonAdjacentDependencies,
    NestedNonAdjacentDependencies,
    NonAdjacentDependencies,
)
from symseq.generators.registry import build, registered_names
from symseq.trial import Trial
from symseq.trial_source import TrialSource

ORDERED_VARIANTS = (CrossedNonAdjacentDependencies, NestedNonAdjacentDependencies)
ALL_VARIANTS = (NonAdjacentDependencies, *ORDERED_VARIANTS)


def _is_simple_grammatical(generator, symbols):
    terminals = dict(generator.dependency_pairs)
    return symbols[0] in terminals and symbols[-1] == terminals[symbols[0]]


def _is_ordered_grammatical(generator, symbols):
    n_deps = generator.n_deps
    start_indices = {symbol: index for index, symbol in enumerate(generator.start_symbols)}
    terminal_indices = {symbol: index for index, symbol in enumerate(generator.terminal_symbols)}
    starts = tuple(start_indices[symbol] for symbol in symbols[:n_deps])
    terminals = tuple(terminal_indices[symbol] for symbol in symbols[n_deps:])
    expected = starts[::-1] if isinstance(generator, NestedNonAdjacentDependencies) else starts
    return terminals == expected


@pytest.mark.parametrize("generator_type", ALL_VARIANTS)
def test_variants_are_registered_exported_trial_sources(generator_type):
    generator = generator_type(n_deps=3, seed=42, verbose=False)

    assert generator_type.__name__ in registered_names()
    assert isinstance(build(generator_type.__name__, n_deps=3, seed=42, verbose=False), generator_type)
    assert isinstance(generator, TrialSource)
    assert all(type(symbol) is str for symbol in generator.alphabet)


@pytest.mark.parametrize("generator_type", ALL_VARIANTS)
def test_custom_dependency_pairs_are_shared_across_variants(generator_type):
    pairs = [(np.str_("left-1"), np.str_("right-1")), (np.str_("left-2"), np.str_("right-2"))]

    generator = generator_type(dependency_pairs=pairs, seed=1, verbose=False)

    assert generator.dependency_pairs == (("left-1", "right-1"), ("left-2", "right-2"))
    assert generator.n_deps == 2
    assert all(type(symbol) is str for symbol in generator.alphabet)


@pytest.mark.parametrize("generator_type", ALL_VARIANTS)
def test_custom_rng_is_used_by_all_variants(generator_type):
    rng = np.random.default_rng(27)

    generator = generator_type(n_deps=2, rng=rng, seed=999, verbose=False)

    assert generator.rng is rng


@pytest.mark.parametrize("generator_type", ALL_VARIANTS)
def test_verbose_false_suppresses_construction_logs(generator_type, caplog):
    with caplog.at_level(logging.INFO):
        generator_type(label="quiet-nad", n_deps=2, seed=1, verbose=False)

    assert "quiet-nad" not in caplog.text


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_deps": 0}, "positive integer"),
        ({"dependency_pairs": []}, "at least one pair"),
        ({"dependency_pairs": [("A", "B", "C")]}, "exactly two symbols"),
        ({"dependency_pairs": [("A", "B"), ("A", "C")]}, "start symbols must be unique"),
        ({"dependency_pairs": [("A", "C"), ("B", "C")]}, "terminal symbols must be unique"),
        ({"dependency_pairs": [("A", "B"), ("B", "C")]}, "must be disjoint"),
        ({"dependency_pairs": [("A", "B")], "n_deps": 2}, "must match"),
        ({"dependency_pairs": [("A", "#")]}, "eos must not overlap"),
    ],
)
def test_common_configuration_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        CrossedNonAdjacentDependencies(**kwargs, verbose=False)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_unique_fillers": -1}, "non-negative integer"),
        ({"fillers": ["X", "X"]}, "fillers must be unique"),
        ({"dependency_pairs": [("A", "B")], "fillers": ["A"]}, "must be disjoint"),
        ({"fillers": ["#"]}, "eos must not overlap"),
    ],
)
def test_simple_filler_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        NonAdjacentDependencies(**kwargs, verbose=False)


def test_simple_default_filler_length_is_one():
    generator = NonAdjacentDependencies(n_deps=2, seed=2, verbose=False)

    symbols = generator.generate_string()
    strings, labels = generator.generate_string_set(3)

    assert len(symbols) == 3
    assert all(len(string) == 3 for string in strings)
    assert labels == [True, True, True]


def test_repeated_and_randomized_filler_vocabularies_are_exact():
    generator = NonAdjacentDependencies(n_deps=2, fillers=("X", "Y"), seed=1, verbose=False)

    repeated = generator.generate_vocabulary(filler_len=2, randomize_fillers=False, verbose=False)
    randomized = generator.generate_vocabulary(filler_len=2, randomize_fillers=True, verbose=False)

    assert len(repeated) == 4
    assert all(string[1] == string[2] for string in repeated)
    assert len(randomized) == 8
    assert {tuple(string[1:3]) for string in randomized} == {("X", "X"), ("X", "Y"), ("Y", "X"), ("Y", "Y")}


def test_empty_fillers_produce_adjacent_dependencies():
    generator = NonAdjacentDependencies(n_deps=2, fillers=(), seed=1, verbose=False)

    symbols = generator.generate_string(filler_len=5, randomize_fillers=True)
    trial = generator.generate_trial(filler_len=5)

    assert len(symbols) == 2
    assert _is_simple_grammatical(generator, symbols)
    assert trial.meta["dependency_length"] == 0


def test_without_replacement_returns_requested_simple_count():
    generator = NonAdjacentDependencies(n_deps=2, fillers=("X", "Y"), seed=3, verbose=False)

    one, one_labels = generator.generate_string_set(1, filler_len=3, replace=False)
    complete, complete_labels = generator.generate_string_set(4, filler_len=3, replace=False)

    assert len(one) == 1
    assert one_labels == [True]
    assert len(complete) == len({tuple(string) for string in complete}) == 4
    assert complete_labels == [True] * 4


def test_randomized_fillers_expand_without_replacement_support():
    generator = NonAdjacentDependencies(n_deps=2, fillers=("X", "Y"), seed=4, verbose=False)

    strings, labels = generator.generate_string_set(
        8,
        filler_len=2,
        randomize_fillers=True,
        replace=False,
    )

    assert len(strings) == len({tuple(string) for string in strings}) == 8
    assert labels == [True] * 8


def test_strict_without_replacement_rejects_oversubscription():
    generator = NonAdjacentDependencies(n_deps=2, fillers=("X", "Y"), seed=1, verbose=False)

    with pytest.raises(ValueError, match="support of size 4"):
        generator.generate_string_set(5, replace=False, strict=True)


def test_nonstrict_without_replacement_returns_complete_support(caplog):
    generator = NonAdjacentDependencies(n_deps=2, fillers=("X", "Y"), seed=1, verbose=False)

    with caplog.at_level(logging.WARNING):
        strings, labels = generator.generate_string_set(10, frac_violations=0.5, replace=False, strict=False)

    assert len(strings) == len({tuple(string) for string in strings}) == 4
    assert Counter(labels) == {True: 2, False: 2}
    assert "returning the complete support" in caplog.text


@pytest.mark.parametrize("generator_type", ORDERED_VARIANTS)
def test_ordered_without_replacement_uses_factorial_support(generator_type):
    generator = generator_type(n_deps=3, seed=2, verbose=False)

    strings, labels = generator.generate_string_set(6, replace=False)

    assert len(strings) == len({tuple(string) for string in strings}) == 6
    assert labels == [True] * 6
    assert all(_is_ordered_grammatical(generator, string) for string in strings)


@pytest.mark.parametrize("generator_type", ORDERED_VARIANTS)
def test_ordered_violations_remain_unique_without_replacement(generator_type):
    generator = generator_type(n_deps=3, seed=6, verbose=False)

    strings, labels = generator.generate_string_set(6, frac_violations=0.5, replace=False)

    assert len(strings) == len({tuple(string) for string in strings}) == 6
    assert labels == [_is_ordered_grammatical(generator, string) for string in strings]


@pytest.mark.parametrize("generator_type", ALL_VARIANTS)
def test_empty_batches_are_supported(generator_type):
    generator = generator_type(n_deps=2, seed=1, verbose=False)

    strings, labels = generator.generate_string_set(0, frac_violations=1.0, replace=False)
    trials = generator.generate_trials(0, frac_violations=1.0, replace=False)

    assert strings == []
    assert labels == []
    assert trials == []


@pytest.mark.parametrize("generator_type", ORDERED_VARIANTS)
def test_ordered_generation_obeys_variant_terminal_order(generator_type):
    generator = generator_type(n_deps=4, seed=9, verbose=False)

    strings, _ = generator.generate_string_set(25)

    assert all(_is_ordered_grammatical(generator, string) for string in strings)


def test_simple_violation_uses_a_mismatched_terminal():
    generator = NonAdjacentDependencies(n_deps=3, fillers=("X",), seed=5, verbose=False)

    symbols = generator.generate_string(violation=True)

    assert not _is_simple_grammatical(generator, symbols)
    assert symbols[-1] in generator.terminal_symbols


@pytest.mark.parametrize("generator_type", ORDERED_VARIANTS)
def test_ordered_violation_preserves_symbols_but_breaks_order(generator_type):
    valid_generator = generator_type(n_deps=4, seed=12, verbose=False)
    invalid_generator = generator_type(n_deps=4, seed=12, verbose=False)

    valid = valid_generator.generate_string()
    invalid = invalid_generator.generate_string(violation=True)

    assert sorted(valid) == sorted(invalid)
    assert _is_ordered_grammatical(valid_generator, valid)
    assert not _is_ordered_grammatical(invalid_generator, invalid)


@pytest.mark.parametrize(
    ("generator", "is_grammatical"),
    [
        (NonAdjacentDependencies(n_deps=3, seed=7, verbose=False), _is_simple_grammatical),
        (CrossedNonAdjacentDependencies(n_deps=3, seed=7, verbose=False), _is_ordered_grammatical),
        (NestedNonAdjacentDependencies(n_deps=3, seed=7, verbose=False), _is_ordered_grammatical),
    ],
)
def test_batch_violation_labels_match_sequence_semantics(generator, is_grammatical):
    strings, grammaticality = generator.generate_string_set(20, frac_violations=0.35)

    assert grammaticality.count(False) == 7
    assert grammaticality == [is_grammatical(generator, string) for string in strings]


@pytest.mark.parametrize("generator_type", ALL_VARIANTS)
def test_full_and_zero_violation_fractions(generator_type):
    generator = generator_type(n_deps=3, seed=8, verbose=False)

    _, grammatical = generator.generate_string_set(5, frac_violations=0.0)
    _, ungrammatical = generator.generate_string_set(5, frac_violations=1.0)

    assert grammatical == [True] * 5
    assert ungrammatical == [False] * 5


@pytest.mark.parametrize("generator_type", ALL_VARIANTS)
def test_one_pair_cannot_generate_requested_violation(generator_type):
    generator = generator_type(n_deps=1, seed=1, verbose=False)

    with pytest.raises(ValueError, match="At least two dependency pairs"):
        generator.generate_string(violation=True)
    with pytest.raises(ValueError, match="At least two dependency pairs"):
        generator.generate_string_set(2, frac_violations=1.0)


@pytest.mark.parametrize("fraction", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_violation_fraction_is_rejected(fraction):
    generator = NonAdjacentDependencies(n_deps=2, seed=1, verbose=False)

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        generator.generate_string_set(2, frac_violations=fraction)


@pytest.mark.parametrize("filler_len", [-1, 1.5, True])
def test_invalid_filler_length_is_rejected(filler_len):
    generator = NonAdjacentDependencies(n_deps=2, seed=1, verbose=False)
    error = TypeError if filler_len in (1.5, True) else ValueError

    with pytest.raises(error, match="filler_len"):
        generator.generate_string(filler_len=filler_len)


def test_simple_violated_trial_has_aligned_targets_and_metadata():
    generator = NonAdjacentDependencies(n_deps=3, seed=11, verbose=False)

    trial = generator.generate_trial(filler_len=2, violation=True)
    pair_index = trial.intrinsic_targets["pair_index"].values

    assert isinstance(trial, Trial)
    assert trial.intrinsic_targets["grammaticality"].values is False
    assert trial.symbols[0] == generator.dependency_pairs[pair_index][0]
    assert trial.symbols[-1] != generator.dependency_pairs[pair_index][1]
    assert trial.meta["grammatical"] is False
    assert trial.meta["dependency_length"] == 2


@pytest.mark.parametrize("generator_type", ORDERED_VARIANTS)
def test_ordered_trial_records_dependency_order(generator_type):
    generator = generator_type(n_deps=3, seed=13, verbose=False)

    trial = generator.generate_trial()
    order = trial.meta["dependency_order"]

    assert trial.intrinsic_targets["grammaticality"].values is True
    assert trial.symbols[:3] == [generator.start_symbols[index] for index in order]
    assert _is_ordered_grammatical(generator, trial.symbols)


@pytest.mark.parametrize("generator_type", ALL_VARIANTS)
def test_trial_batches_have_exact_violation_fraction(generator_type):
    generator = generator_type(n_deps=3, seed=14, verbose=False)

    trials = generator.generate_trials(7, frac_violations=0.5)
    labels = [trial.intrinsic_targets["grammaticality"].values for trial in trials]

    assert len(trials) == 7
    assert labels.count(False) == 3
    assert all(trial.meta["grammatical"] is label for trial, label in zip(trials, labels, strict=True))


@pytest.mark.parametrize("generator_type", ALL_VARIANTS)
def test_draw_trial_and_batch_work_for_all_variants(generator_type):
    generator = generator_type(n_deps=3, seed=15, verbose=False)

    trial = generator.draw_trial(violation=True)
    batch = generator.draw_batch(4, frac_violations=0.5)

    assert isinstance(trial, Trial)
    assert trial.intrinsic_targets["grammaticality"].values is False
    assert len(batch) == 4
    assert sum(not item.intrinsic_targets["grammaticality"].values for item in batch) == 2


@pytest.mark.parametrize("generator_type", ALL_VARIANTS)
def test_seed_reproducibility_for_all_variants(generator_type):
    first = generator_type(n_deps=3, seed=2026, verbose=False)
    second = generator_type(n_deps=3, seed=2026, verbose=False)

    first_strings, first_labels = first.generate_string_set(12, frac_violations=0.25)
    second_strings, second_labels = second.generate_string_set(12, frac_violations=0.25)

    assert first_strings == second_strings
    assert first_labels == second_labels
