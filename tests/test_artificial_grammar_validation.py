"""Regression tests for ArtificialGrammar validation and generation boundaries."""

import numpy as np
import pytest

import symseq.generators.ag.artificial_grammar as artificial_grammar_module
from symseq.generators.ag import ArtificialGrammar


def _grammar(**overrides):
    parameters = {
        "label": "validation-test",
        "states": ["A", "B"],
        "alphabet": ["A", "B"],
        "transitions": [("A", "B", 1.0)],
        "start_states": ["A"],
        "terminal_states": ["B"],
        "seed": 7,
    }
    parameters.update(overrides)
    return ArtificialGrammar(**parameters)


def test_generate_string_allows_one_successful_attempt():
    grammar = _grammar()

    assert grammar.generate_string(max_iter=1) == ["A", "B"]


@pytest.mark.parametrize(
    "generation_kwargs",
    [
        {"min_length": 1.0},
        {"max_length": 2.0},
        {"length_range": (1.0, 2.0)},
        {"length_range": (1, 2.0)},
    ],
)
def test_generate_string_rejects_float_length_bounds(generation_kwargs):
    grammar = _grammar()

    with pytest.raises(ValueError, match="integers"):
        grammar.generate_string(**generation_kwargs)


def test_generate_string_accepts_integer_list_length_range():
    grammar = _grammar()

    assert grammar.generate_string(length_range=[2, 2], max_iter=1) == ["A", "B"]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"start_states": ["missing"]}, "start_states"),
        ({"terminal_states": ["missing"]}, "terminal_states"),
        ({"states": ["A", "A", "B"]}, "duplicate"),
        ({"terminal_states": ["#"]}, "EOS"),
    ],
)
def test_constructor_rejects_invalid_state_declarations(overrides, message):
    with pytest.raises(ValueError, match=message):
        _grammar(**overrides)


@pytest.mark.parametrize(
    "transitions",
    [
        [("A", "B", -0.1), ("A", "A", 1.1)],
        [("A", "B", np.nan)],
        [("A", "B", np.inf)],
        [("A", "B", 0.75)],
    ],
)
def test_constructor_rejects_invalid_transition_probabilities(transitions):
    with pytest.raises(ValueError):
        _grammar(transitions=transitions)


def test_constructor_rejects_duplicate_transition_edges():
    transitions = [("A", "B", 0.5), ("A", "B", 0.5)]

    with pytest.raises(ValueError, match="duplicate"):
        _grammar(transitions=transitions)


def test_constructor_rejects_unknown_transition_states():
    with pytest.raises(ValueError, match="unknown state"):
        _grammar(transitions=[("A", "missing", 1.0)])


def test_constructor_does_not_mutate_supplied_transitions():
    transitions = [("A", "B", 1.0)]
    original = transitions.copy()

    grammar = _grammar(transitions=transitions)

    assert transitions == original
    assert ("B", grammar.eos, 1.0) in grammar.transitions


def test_construction_and_default_table_access_do_not_require_pandas(monkeypatch):
    def fail_if_called():
        raise AssertionError("pandas should not be imported")

    monkeypatch.setattr(artificial_grammar_module, "_require_pandas", fail_if_called)

    grammar = _grammar()
    verbose_grammar = _grammar(verbose=True)

    np.testing.assert_array_equal(grammar.get_transition_table(), grammar.transition_table)
    assert verbose_grammar.label == grammar.label
