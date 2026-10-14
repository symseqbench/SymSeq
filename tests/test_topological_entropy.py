"""Tests for topological entropy validation."""

import warnings

import numpy as np
import pytest

from symseq.metrics.grammar.topological import TopologicalEntropyWarning, topological_entropy


def test_lift_requires_sequence():
    with pytest.raises(ValueError, match="requires a sequence"):
        topological_entropy(method="lift", verbose=False)


def test_direct_requires_numpy_array():
    with pytest.raises(TypeError, match="Binary transition table"):
        topological_entropy(transitions=[[0, 1], [1, 0]], method="direct", verbose=False)


def test_direct_requires_square_2d_transition_table():
    transitions = np.array([[0, 1, 0], [1, 0, 1]])
    with pytest.raises(ValueError, match="square 2D"):
        topological_entropy(transitions=transitions, method="direct", verbose=False)


def test_direct_requires_binary_transition_table():
    transitions = np.array([[0.0, 0.5], [1.0, 0.0]])
    with pytest.raises(ValueError, match="binary transition table"):
        topological_entropy(transitions=transitions, method="direct", verbose=False)


# --- Grammar-level TE and the loop-back (EOS -> start) convention ------------------------------------------------
# Reference values: TE of the transition lists in Vokey & Jamieson (2020), Table 1, which transcribe the 10 grammars of
# Schiff & Katan (2014), Fig. 1 (computed with their computeTE; closed graph). The presets generate the same strings.
# Schiff & Katan's own values differ for three grammars: A (0.560; reproduced by dropping the P->N edge), F (0.716;
# obtained by collapsing the final M into the loop-back, Vokey & Jamieson p. 358) and G (0.740; reproduced by
# merging the parallel T and exit edges from S1 to S0).
REFERENCE_TE = [
    ("FSG_G1_A", 0.5882),
    ("FSG_G2_B", 0.5778),
    ("FSG_G3_C", 0.6024),
    ("FSG_G4_D", 0.6033),
    ("FSG_G5_E", 0.6858),
    ("FSG_G6_F", 0.6563),
    ("FSG_G7_G", 0.7828),
    ("FSG_G8_H", 0.7608),
    ("FSG_G9_I", 0.8563),
    ("FSG_G10_J", 0.9161),
]

# Acyclic grammar: A followed by B or C, then EOS. Closed, it has two 3-cycles through EOS, so TE = ln(2) / 3.
ACYCLIC = {
    "label": "acyclic",
    "states": ["#", "A(1)", "B(1)", "C(1)"],
    "alphabet": ["A", "B", "C"],
    "start_states": ["A(1)"],
    "terminal_states": ["B(1)", "C(1)"],
    "transitions": [("A(1)", "B(1)", 0.5), ("A(1)", "C(1)", 0.5), ("B(1)", "#", 1.0), ("C(1)", "#", 1.0)],
    "eos": "#",
}


def _grammar(preset_name):
    from symseq.generators.ag.artificial_grammar import ArtificialGrammar
    from symseq.generators.presets import ag as presets

    return ArtificialGrammar(**getattr(presets, preset_name), verbose=False)


@pytest.mark.parametrize(("preset_name", "expected"), REFERENCE_TE)
def test_closed_grammar_matches_published_te(preset_name, expected):
    grammar = _grammar(preset_name)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        te = grammar.topological_entropy(correct_terminal_sink=True)
    assert te == pytest.approx(expected, abs=1e-3)


def test_open_grammar_warns_and_differs_from_closed():
    grammar = _grammar("FSG_G3_C")
    with pytest.warns(TopologicalEntropyWarning, match="OPEN GRAMMAR"):
        open_te = grammar.topological_entropy()
    assert open_te == pytest.approx(0.3822, abs=1e-3)


def test_acyclic_grammar_warns_open_and_acyclic_but_is_positive_when_closed():
    from symseq.generators.ag.artificial_grammar import ArtificialGrammar

    grammar = ArtificialGrammar(**ACYCLIC, verbose=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert grammar.topological_entropy() == 0.0
    messages = [str(w.message) for w in caught if w.category is TopologicalEntropyWarning]
    assert any("OPEN GRAMMAR" in m for m in messages)
    assert any("ACYCLIC" in m for m in messages)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert grammar.topological_entropy(correct_terminal_sink=True) == pytest.approx(np.log(2) / 3)


def test_open_grammar_warning_names_the_unlinked_start_states():
    grammar = _grammar("FSG_G3_C")
    with pytest.warns(TopologicalEntropyWarning, match=r"\['T\(1\)', 'V\(1\)'\]"):
        grammar.topological_entropy()


def test_synthesized_grammar_is_already_closed():
    from symseq.generators.ag.artificial_grammar import ArtificialGrammar

    grammar = ArtificialGrammar.from_constraints(label="synth", alphabet_size=4, target_complexity=0.5, seed=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        grammar.topological_entropy()
    assert not any("OPEN GRAMMAR" in str(w.message) for w in caught)


def test_binary_transition_table_defaults_to_open():
    grammar = _grammar("FSG_G3_C")
    np.testing.assert_array_equal(grammar.get_binary_transition_table(), grammar.get_transition_table(binary=True))
