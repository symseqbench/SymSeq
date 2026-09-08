"""Tests for Markov and Context-tree ArtificialGrammar inference."""

import numpy as np
import pytest

from symseq.generators.ag import ArtificialGrammar
from symseq.generators.ag.inference import infer_markov, infer_vlmc
from symseq.metrics.grammar.markov import vlmc_fit


def _state_for_context(grammar, context):
    state_contexts = grammar.metadata["inference"]["state_contexts"]
    return next(state for state, candidate in state_contexts.items() if candidate == context)


def _transition_probability(grammar, source, target):
    return next(probability for src, tgt, probability in grammar.transitions if src == source and tgt == target)


def test_markov_inference_preserves_boundaries_and_start_frequencies():
    sequences = [["A", "B"], ["A", "B"], ["B", "A"]]

    grammar = infer_markov(sequences, order=1, seed=7)

    assert isinstance(grammar, ArtificialGrammar)
    assert grammar.start_probabilities == pytest.approx({"A": 2 / 3, "B": 1 / 3})
    assert _transition_probability(grammar, "A", "B") == pytest.approx(2 / 3)
    assert _transition_probability(grammar, "A", grammar.eos) == pytest.approx(1 / 3)
    assert _transition_probability(grammar, "B", "A") == pytest.approx(1 / 3)
    assert _transition_probability(grammar, "B", grammar.eos) == pytest.approx(2 / 3)


def test_higher_order_markov_uses_indexed_context_states():
    sequences = [["A", "B", "X"]] * 20 + [["C", "B", "Y"]] * 20

    grammar = infer_markov(sequences, order=2)
    ab_state = _state_for_context(grammar, ("A", "B"))
    cb_state = _state_for_context(grammar, ("C", "B"))
    bx_state = _state_for_context(grammar, ("B", "X"))
    by_state = _state_for_context(grammar, ("B", "Y"))

    assert ab_state != cb_state
    assert _transition_probability(grammar, ab_state, bx_state) == 1.0
    assert _transition_probability(grammar, cb_state, by_state) == 1.0


def test_higher_order_recognition_tracks_reachable_indexed_states():
    sequences = [["A", "B", "X"]] * 20 + [["C", "B", "Y"]] * 20
    grammar = infer_markov(sequences, order=2)

    assert grammar.is_grammatical(["A", "B", "X"])
    assert grammar.is_grammatical(["C", "B", "Y", grammar.eos])
    assert not grammar.is_grammatical(["A", "B", "Y"])
    assert not grammar.is_grammatical(["C", "B", "X", grammar.eos])


def test_higher_order_recognition_honors_explicit_state_paths():
    sequences = [["A", "B", "X"]] * 20 + [["C", "B", "Y"]] * 20
    grammar = infer_markov(sequences, order=2)
    ab_state = _state_for_context(grammar, ("A", "B"))
    cb_state = _state_for_context(grammar, ("C", "B"))
    bx_state = _state_for_context(grammar, ("B", "X"))
    by_state = _state_for_context(grammar, ("B", "Y"))

    assert grammar.is_grammatical(["A", ab_state, bx_state])
    assert grammar.is_grammatical(["C", cb_state, by_state, grammar.eos])
    assert not grammar.is_grammatical(["A", cb_state, by_state])


def test_recognition_requires_a_complete_sequence():
    grammar = infer_markov([["A", "B", "X"]], order=2)

    assert not grammar.is_grammatical([])
    assert not grammar.is_grammatical(["A", "B"])
    assert not grammar.is_grammatical(["A", grammar.eos, "B", "X"])
    assert not grammar.is_grammatical(["A", "B", "X", grammar.eos, grammar.eos])


def test_markov_bic_selection_is_reported():
    sequences = [["A", "B"] * 20 for _ in range(10)]

    grammar = infer_markov(sequences, order="bic", max_order=3)
    inference = grammar.metadata["inference"]

    assert inference["criterion"] == "BIC"
    assert inference["order"] in range(4)
    assert set(inference["bic_scores"]) == set(range(4))


def test_context_vlmc_retains_predictive_long_contexts():
    sequences = [["A", "B", "X"]] * 100 + [["C", "B", "Y"]] * 100

    grammar = infer_vlmc(sequences, max_depth=2, min_count=2, alpha=0.05)
    contexts = set(grammar.metadata["inference"]["contexts"])

    assert ("A", "B") in contexts
    assert ("C", "B") in contexts
    ab_state = _state_for_context(grammar, ("A", "B"))
    cb_state = _state_for_context(grammar, ("C", "B"))
    assert _transition_probability(grammar, ab_state, "X") == 1.0
    assert _transition_probability(grammar, cb_state, "Y") == 1.0


def test_context_vlmc_prunes_uninformative_extensions():
    sequences = [["A", "B", "X"]] * 100 + [["C", "B", "X"]] * 100

    grammar = infer_vlmc(sequences, max_depth=2, min_count=2)
    contexts = set(grammar.metadata["inference"]["contexts"])

    assert ("A", "B") not in contexts
    assert ("C", "B") not in contexts
    assert ("B",) in contexts


def test_artificial_grammar_from_sequences_dispatches_backends():
    sequences = [["A", "B"], ["A", "B"]]

    markov = ArtificialGrammar.from_sequences(sequences, method="markov", order=1)
    vlmc = ArtificialGrammar.from_sequences(sequences, method="vlmc", max_depth=2)

    assert markov.metadata["inference"]["method"] == "markov"
    assert vlmc.metadata["inference"]["method"] == "vlmc_context"


def test_legacy_vlmc_fit_behavior_is_unchanged():
    result = vlmc_fit(
        ["A", "B", "A", "B"],
        max_depth=2,
        min_count=1,
        pruning_threshold=0.01,
    )

    assert set(result["context_tree"]) == {
        ("A",),
        ("B",),
        ("A", "B"),
        ("B", "A"),
    }


def test_inferred_start_probabilities_control_sampling():
    sequences = [["A"]] * 9 + [["B"]]
    grammar = infer_markov(sequences, order=1, seed=11)

    starts = [grammar.generate_string(max_length=1)[0] for _ in range(200)]

    assert starts.count("A") > starts.count("B")


@pytest.mark.parametrize("method", ["unknown", "scfg"])
def test_from_sequences_rejects_unknown_method(method):
    with pytest.raises(ValueError, match="method must be"):
        ArtificialGrammar.from_sequences([["A"]], method=method)


def test_inference_rejects_eos_inside_corpus():
    with pytest.raises(ValueError, match="must not contain"):
        infer_markov([["A", "#"]])


def test_uniform_start_sampling_path_remains_reproducible():
    first = ArtificialGrammar.from_preset("Elman", seed=123)
    second = ArtificialGrammar.from_preset("Elman", seed=123)

    assert first.generate_string() == second.generate_string()
    assert np.isclose(sum(first.start_probabilities.values()), 1.0)
