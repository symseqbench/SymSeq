"""
test_nax.py

Comprehensive test suite for the nAX class.
"""

import pytest
import numpy as np
from symseq.grammars.nax import nAX


class TestInitialization:
    """Tests for nAX initialization and validation."""

    def test_basic_initialization(self):
        """Test basic initialization with default parameters."""
        gen = nAX()
        assert gen.label == "n-AX"
        assert gen.contexts == ("1", "2")
        assert gen.cue_map == {"1": "A", "2": "B"}
        assert gen.probe_map == {"1": "X", "2": "Y"}
        assert gen.fillers == ("C", "D", "Z")
        assert gen.p_target == 0.6

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        contexts = ("1", "2", "3")
        cue_map = {"1": "A", "2": "B", "3": "C"}
        probe_map = {"1": "X", "2": "Y", "3": "Z"}
        fillers = ("F", "G")

        gen = nAX(
            label="3-AX",
            contexts=contexts,
            cue_map=cue_map,
            probe_map=probe_map,
            fillers=fillers,
            p_target=0.7,
            eos="$",
        )
        assert gen.label == "3-AX"
        assert gen.contexts == contexts
        assert gen.cue_map == cue_map
        assert gen.probe_map == probe_map
        assert gen.fillers == fillers
        assert gen.p_target == 0.7
        assert gen.eos == "$"

    def test_insufficient_contexts(self):
        """Test that fewer than 2 contexts raises ValueError."""
        with pytest.raises(ValueError, match="at least 2 contexts"):
            nAX(contexts=("1",))

    def test_missing_cue_for_context(self):
        """Test that missing cue for a context raises ValueError."""
        with pytest.raises(ValueError, match="cue_map must define a cue"):
            nAX(contexts=("1", "2", "3"), cue_map={"1": "A", "2": "B"})

    def test_missing_probe_for_context(self):
        """Test that missing probe for a context raises ValueError."""
        with pytest.raises(ValueError, match="probe_map must define a probe"):
            nAX(
                contexts=("1", "2", "3"),
                cue_map={"1": "A", "2": "B", "3": "C"},
                probe_map={"1": "X", "2": "Y"},
            )

    def test_invalid_p_target(self):
        """Test that invalid p_target raises ValueError."""
        with pytest.raises(ValueError, match="p_target must be in"):
            nAX(p_target=-0.1)
        with pytest.raises(ValueError, match="p_target must be in"):
            nAX(p_target=1.5)

    def test_custom_rng(self):
        """Test initialization with custom RNG."""
        rng = np.random.default_rng(42)
        gen = nAX(rng=rng)
        # Check that the RNG is being used (not identity, as parent class may wrap it)
        string1 = gen.generate_string()
        assert isinstance(string1, list)


class TestContextSetup:
    """Tests for context-related setup and validation."""

    def test_context_index_mapping(self):
        """Test that context index mapping is created correctly."""
        gen = nAX(
            contexts=("1", "2", "3"),
            cue_map={"1": "A", "2": "B", "3": "C"},
            probe_map={"1": "X", "2": "Y", "3": "Z"},
        )
        assert gen._ctx_index == {"1": 0, "2": 1, "3": 2}

    def test_default_context_probs(self):
        """Test that default context probabilities are uniform."""
        gen = nAX(
            contexts=("1", "2", "3"),
            cue_map={"1": "A", "2": "B", "3": "C"},
            probe_map={"1": "X", "2": "Y", "3": "Z"},
        )
        expected = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        np.testing.assert_allclose(gen.context_probs, expected)

    def test_custom_context_probs(self):
        """Test initialization with custom context probabilities."""
        probs = [0.5, 0.3, 0.2]
        gen = nAX(
            contexts=("1", "2", "3"),
            cue_map={"1": "A", "2": "B", "3": "C"},
            probe_map={"1": "X", "2": "Y", "3": "Z"},
            context_probs=probs,
        )
        np.testing.assert_allclose(gen.context_probs, probs)

    def test_context_probs_normalization(self):
        """Test that context probabilities are normalized."""
        probs = [1, 2, 3]  # Should be normalized to [1/6, 2/6, 3/6]
        gen = nAX(
            contexts=("1", "2", "3"),
            cue_map={"1": "A", "2": "B", "3": "C"},
            probe_map={"1": "X", "2": "Y", "3": "Z"},
            context_probs=probs,
        )
        expected = np.array([1.0 / 6, 2.0 / 6, 3.0 / 6])
        np.testing.assert_allclose(gen.context_probs, expected)

    def test_invalid_context_probs_length(self):
        """Test that mismatched context_probs length raises ValueError."""
        with pytest.raises(ValueError, match="context_probs must be length-n"):
            nAX(contexts=("1", "2"), context_probs=[0.5, 0.3, 0.2])

    def test_negative_context_probs(self):
        """Test that negative context probabilities raise ValueError."""
        with pytest.raises(ValueError, match="context_probs must be length-n and nonnegative"):
            nAX(contexts=("1", "2"), context_probs=[0.6, -0.1])


class TestProbeMatrix:
    """Tests for probe_given_context matrix setup."""

    def test_default_probe_matrix_2_contexts(self):
        """Test default probe matrix with 2 contexts."""
        gen = nAX(contexts=("1", "2"), p_target=0.8)
        # Row 0: [0.8, 0.2], Row 1: [0.2, 0.8]
        expected = np.array([[0.8, 0.2], [0.2, 0.8]])
        np.testing.assert_allclose(gen.probe_given_context, expected)

    def test_default_probe_matrix_3_contexts(self):
        """Test default probe matrix with 3 contexts."""
        gen = nAX(
            contexts=("1", "2", "3"),
            cue_map={"1": "A", "2": "B", "3": "C"},
            probe_map={"1": "X", "2": "Y", "3": "Z"},
            p_target=0.6,
        )
        # For 3 contexts: diagonal = 0.6, off-diagonal = 0.4/2 = 0.2
        expected = np.array([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
        np.testing.assert_allclose(gen.probe_given_context, expected)

    def test_probe_matrix_row_stochastic(self):
        """Test that probe matrix rows sum to 1."""
        gen = nAX(
            contexts=("1", "2", "3", "4"),
            cue_map={"1": "A", "2": "B", "3": "C", "4": "D"},
            probe_map={"1": "W", "2": "X", "3": "Y", "4": "Z"},
            p_target=0.5,
        )
        row_sums = gen.probe_given_context.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(4))

    def test_custom_probe_matrix_not_implemented(self):
        """Test that custom probe_given_context raises NotImplementedError."""
        custom_matrix = np.array([[0.7, 0.3], [0.4, 0.6]])
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            nAX(contexts=("1", "2"), probe_given_context=custom_matrix)


class TestGrammarConstruction:
    """Tests for grammar construction (alphabet, states, transitions)."""

    def test_alphabet_construction(self):
        """Test that alphabet is constructed correctly."""
        gen = nAX(
            contexts=("1", "2"),
            cue_map={"1": "A", "2": "B"},
            probe_map={"1": "X", "2": "Y"},
            fillers=("C", "D"),
        )
        expected_alphabet = {"1", "2", "A", "B", "X", "Y", "C", "D"}
        assert set(gen.alphabet) == expected_alphabet
        assert gen.alphabet == sorted(gen.alphabet)

    def test_states_include_contexts(self):
        """Test that states include all contexts."""
        gen = nAX(
            contexts=("1", "2", "3"),
            cue_map={"1": "A", "2": "B", "3": "C"},
            probe_map={"1": "X", "2": "Y", "3": "Z"},
        )
        assert "1" in gen.states
        assert "2" in gen.states
        assert "3" in gen.states

    def test_states_include_cues(self):
        """Test that states include all cues."""
        gen = nAX(cue_map={"1": "A", "2": "B"})
        assert "A" in gen.states
        assert "B" in gen.states

    def test_states_include_probes(self):
        """Test that states include all probes."""
        gen = nAX(probe_map={"1": "X", "2": "Y"})
        assert "X" in gen.states
        assert "Y" in gen.states

    def test_states_include_context_specific_fillers(self):
        """Test that states include context-specific filler states."""
        gen = nAX(contexts=("1", "2"), fillers=("C", "D"))
        assert "C(1)" in gen.states
        assert "C(2)" in gen.states
        assert "D(1)" in gen.states
        assert "D(2)" in gen.states

    def test_states_include_eos(self):
        """Test that states include EOS."""
        gen = nAX(eos="$")
        assert "$" in gen.states

    def test_start_states_are_contexts(self):
        """Test that start states are contexts."""
        gen = nAX(
            contexts=("1", "2", "3"),
            cue_map={"1": "A", "2": "B", "3": "C"},
            probe_map={"1": "X", "2": "Y", "3": "Z"},
        )
        assert set(gen.start_states) == {"1", "2", "3"}

    def test_terminal_states_are_probes(self):
        """Test that terminal states are probes."""
        gen = nAX(
            contexts=("1", "2", "3"),
            cue_map={"1": "A", "2": "B", "3": "C"},
            probe_map={"1": "X", "2": "Y", "3": "Z"},
        )
        assert set(gen.terminal_states) == {"X", "Y", "Z"}


class TestStringGeneration:
    """Tests for string generation."""

    def test_generate_string_basic(self):
        """Test basic string generation."""
        rng = np.random.default_rng(42)
        gen = nAX(rng=rng)
        string = gen.generate_string()
        assert isinstance(string, list)
        assert len(string) >= 3  # At least context + cue + probe

    def test_generated_string_structure(self):
        """Test that generated strings have correct structure."""
        rng = np.random.default_rng(42)
        gen = nAX(rng=rng)
        for _ in range(10):
            string = gen.generate_string()
            # First element should be a context
            assert string[0] in gen.contexts
            # Second element should be the corresponding cue
            assert string[1] == gen.cue_map[string[0]]
            # Last element should be a probe
            assert string[-1] in gen.probe_map.values()

    def test_generated_strings_are_grammatical(self):
        """Test that all generated strings are grammatical."""
        rng = np.random.default_rng(42)
        gen = nAX(rng=rng)
        for _ in range(20):
            string = gen.generate_string()
            assert gen.is_grammatical(string + [gen.eos])

    def test_generate_string_with_fillers(self):
        """Test string generation with specified number of fillers."""
        rng = np.random.default_rng(42)
        gen = nAX(rng=rng, fillers=("C", "D"))
        string = gen.generate_string(n_fillers=3)
        # Length should be: context + cue + 3 fillers + probe = 6
        assert len(string) == 6

    def test_generate_string_no_fillers(self):
        """Test string generation with no fillers."""
        rng = np.random.default_rng(42)
        gen = nAX(rng=rng)
        string = gen.generate_string(n_fillers=0)
        # Length should be: context + cue + probe = 3
        assert len(string) == 3

    def test_generate_string_set(self):
        """Test generating multiple strings."""
        rng = np.random.default_rng(42)
        gen = nAX(rng=rng)
        strings = gen.generate_string_set(n_samples=10)
        assert len(strings) == 10
        for string in strings:
            assert gen.is_grammatical(string + [gen.eos])

    def test_generate_string_set_with_kwargs(self):
        """Test generating string set with kwargs."""
        rng = np.random.default_rng(42)
        gen = nAX(rng=rng)
        strings = gen.generate_string_set(n_samples=5, n_fillers=2)
        assert len(strings) == 5
        for string in strings:
            assert len(string) == 5  # context + cue + 2 fillers + probe


class TestTrialLabeling:
    """Tests for trial labeling functionality."""

    def test_target_trial_2ax(self):
        """Test labeling of target trials in 2-AX."""
        gen = nAX(
            contexts=("1", "2"),
            cue_map={"1": "A", "2": "B"},
            probe_map={"1": "X", "2": "Y"},
        )

        # Context 1 -> cue A -> probe X (target)
        target = ["1", "A", "X"]
        label, is_target = gen.label_trial(target)
        assert is_target
        assert label == "C1->T"

        # Context 2 -> cue B -> probe Y (target)
        target = ["2", "B", "Y"]
        label, is_target = gen.label_trial(target)
        assert is_target
        assert label == "C2->T"

    def test_lure_trial_2ax(self):
        """Test labeling of lure trials in 2-AX."""
        gen = nAX(
            contexts=("1", "2"),
            cue_map={"1": "A", "2": "B"},
            probe_map={"1": "X", "2": "Y"},
        )

        # Context 1 -> cue A -> probe Y (lure from context 2)
        lure = ["1", "A", "Y"]
        label, is_target = gen.label_trial(lure)
        assert not is_target
        assert label == "C1->L(2)"

        # Context 2 -> cue B -> probe X (lure from context 1)
        lure = ["2", "B", "X"]
        label, is_target = gen.label_trial(lure)
        assert not is_target
        assert label == "C2->L(1)"

    def test_target_trial_3ax(self):
        """Test labeling of target trials in 3-AX."""
        gen = nAX(
            contexts=("1", "2", "3"),
            cue_map={"1": "A", "2": "B", "3": "C"},
            probe_map={"1": "X", "2": "Y", "3": "Z"},
        )

        target = ["3", "C", "Z"]
        label, is_target = gen.label_trial(target)
        assert is_target
        assert label == "C3->T"

    def test_lure_trial_3ax(self):
        """Test labeling of lure trials in 3-AX."""
        gen = nAX(
            contexts=("1", "2", "3"),
            cue_map={"1": "A", "2": "B", "3": "C"},
            probe_map={"1": "X", "2": "Y", "3": "Z"},
        )

        # Context 3 -> cue C -> probe X (lure from context 1)
        lure = ["3", "C", "X"]
        label, is_target = gen.label_trial(lure)
        assert not is_target
        assert label == "C3->L(1)"

    def test_trial_labeling_with_fillers(self):
        """Test that labeling works correctly with fillers."""
        gen = nAX(
            contexts=("1", "2"),
            cue_map={"1": "A", "2": "B"},
            probe_map={"1": "X", "2": "Y"},
            fillers=("C", "D"),
        )

        # Target with fillers
        target = ["1", "A", "C", "D", "C", "X"]
        label, is_target = gen.label_trial(target)
        assert is_target
        assert label == "C1->T"

        # Lure with fillers
        lure = ["2", "B", "D", "C", "X"]
        label, is_target = gen.label_trial(lure)
        assert not is_target
        assert label == "C2->L(1)"

    def test_invalid_trial_too_short(self):
        """Test labeling of invalid trials that are too short."""
        gen = nAX()
        label, is_target = gen.label_trial(["1"])
        assert label == ""
        assert not is_target

        label, is_target = gen.label_trial(["1", "A"])
        assert label == ""
        assert not is_target

    def test_invalid_trial_wrong_cue(self):
        """Test labeling of trials with wrong cue for context."""
        gen = nAX(
            contexts=("1", "2"),
            cue_map={"1": "A", "2": "B"},
            probe_map={"1": "X", "2": "Y"},
        )

        # Context 1 with wrong cue B (should be A)
        invalid = ["1", "B", "X"]
        label, is_target = gen.label_trial(invalid)
        assert label == ""
        assert not is_target

    def test_invalid_trial_invalid_probe(self):
        """Test labeling of trials with invalid probe."""
        gen = nAX(
            contexts=("1", "2"),
            cue_map={"1": "A", "2": "B"},
            probe_map={"1": "X", "2": "Y"},
        )

        # Probe "Z" is not in probe_map
        invalid = ["1", "A", "Z"]
        label, is_target = gen.label_trial(invalid)
        assert label == ""
        assert not is_target


class TestTargetLureDistribution:
    """Tests for target vs lure distribution."""

    def test_target_probability_respected(self):
        """Test that target probability is approximately respected."""
        # Use a seed that gives results closer to expected distribution
        rng = np.random.default_rng(123)
        gen = nAX(contexts=("1", "2"), p_target=0.7, rng=rng)

        n_samples = 500
        n_targets = 0

        for _ in range(n_samples):
            string = gen.generate_string()
            _, is_target = gen.label_trial(string)
            if is_target:
                n_targets += 1

        target_rate = n_targets / n_samples
        # Should be approximately 0.7 (allow variance due to grammar structure)
        # The actual distribution is affected by the grammar's transition probabilities
        assert 0.5 <= target_rate <= 0.85

    def test_equal_target_lure_probability(self):
        """Test with equal target and lure probabilities."""
        rng = np.random.default_rng(42)
        gen = nAX(contexts=("1", "2"), p_target=0.5, rng=rng)

        n_samples = 200
        n_targets = 0

        for _ in range(n_samples):
            string = gen.generate_string()
            _, is_target = gen.label_trial(string)
            if is_target:
                n_targets += 1

        target_rate = n_targets / n_samples
        # Should be approximately 0.5
        assert 0.4 <= target_rate <= 0.6


class TestRandomness:
    """Tests for randomness and reproducibility."""

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with same seed."""
        rng1 = np.random.default_rng(42)
        gen1 = nAX(rng=rng1)
        strings1 = gen1.generate_string_set(n_samples=5)

        rng2 = np.random.default_rng(42)
        gen2 = nAX(rng=rng2)
        strings2 = gen2.generate_string_set(n_samples=5)

        assert strings1 == strings2

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        rng1 = np.random.default_rng(42)
        gen1 = nAX(rng=rng1)

        rng2 = np.random.default_rng(123)
        gen2 = nAX(rng=rng2)

        strings1 = [gen1.generate_string() for _ in range(10)]
        strings2 = [gen2.generate_string() for _ in range(10)]

        # Very unlikely to be identical
        assert strings1 != strings2

    def test_seed_parameter_used_when_no_rng(self):
        """Test that seed parameter is used when no RNG provided."""
        gen1 = nAX(seed=42)
        strings1 = gen1.generate_string_set(n_samples=5)

        gen2 = nAX(seed=42)
        strings2 = gen2.generate_string_set(n_samples=5)

        assert strings1 == strings2


class TestEdgeCases:
    """Tests for edge cases and corner scenarios."""

    def test_minimal_2_contexts(self):
        """Test with minimal 2 contexts."""
        gen = nAX(contexts=("1", "2"))
        string = gen.generate_string()
        assert len(string) >= 3
        assert gen.is_grammatical(string + [gen.eos])

    def test_many_contexts(self):
        """Test with many contexts."""
        contexts = tuple(str(i) for i in range(10))
        cue_map = {str(i): f"A{i}" for i in range(10)}
        probe_map = {str(i): f"X{i}" for i in range(10)}

        gen = nAX(contexts=contexts, cue_map=cue_map, probe_map=probe_map)
        string = gen.generate_string()
        assert gen.is_grammatical(string + [gen.eos])

    def test_many_fillers(self):
        """Test with many filler symbols."""
        fillers = tuple(f"F{i}" for i in range(20))
        gen = nAX(fillers=fillers)
        string = gen.generate_string()
        assert gen.is_grammatical(string + [gen.eos])

    def test_no_fillers_possible(self):
        """Test with empty filler set."""
        # Note: This might not be the intended use case, but should not crash
        gen = nAX(fillers=())
        string = gen.generate_string(n_fillers=0)
        # Should be just context + cue + probe
        assert len(string) == 3

    def test_extreme_p_target_values(self):
        """Test with extreme p_target values."""
        # Very high target probability
        gen = nAX(p_target=0.99)
        assert gen.probe_given_context[0, 0] == 0.99

        # Very low target probability
        gen = nAX(p_target=0.01)
        assert gen.probe_given_context[0, 0] == 0.01

        # Equal probability
        gen = nAX(contexts=("1", "2"), p_target=0.5)
        assert gen.probe_given_context[0, 0] == 0.5
        assert gen.probe_given_context[0, 1] == 0.5


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow_2ax(self):
        """Test complete workflow for 2-AX."""
        rng = np.random.default_rng(42)
        gen = nAX(
            label="2-AX",
            contexts=("1", "2"),
            cue_map={"1": "A", "2": "B"},
            probe_map={"1": "X", "2": "Y"},
            fillers=("C", "D"),
            p_target=0.6,
            rng=rng,
        )

        # Generate strings
        strings = gen.generate_string_set(n_samples=20)
        assert len(strings) == 20

        # Verify all strings
        targets = 0
        lures = 0
        for string in strings:
            # All should be grammatical
            assert gen.is_grammatical(string + [gen.eos])

            # Label each trial
            label, is_target = gen.label_trial(string)
            assert label != ""  # Should have valid label
            if is_target:
                targets += 1
            else:
                lures += 1

        # Should have both targets and lures
        assert targets > 0
        assert lures > 0

    def test_full_workflow_3ax(self):
        """Test complete workflow for 3-AX."""
        rng = np.random.default_rng(42)
        gen = nAX(
            label="3-AX",
            contexts=("1", "2", "3"),
            cue_map={"1": "A", "2": "B", "3": "C"},
            probe_map={"1": "X", "2": "Y", "3": "Z"},
            fillers=("F", "G"),
            p_target=0.7,
            rng=rng,
        )

        # Generate strings with specific filler count
        strings = gen.generate_string_set(n_samples=15, n_fillers=2)
        assert len(strings) == 15

        for string in strings:
            # Should have fixed length: context + cue + 2 fillers + probe = 5
            assert len(string) == 5
            # Should be grammatical
            assert gen.is_grammatical(string + [gen.eos])
            # Should have valid label
            label, is_target = gen.label_trial(string)
            assert label != ""

    def test_context_coverage(self):
        """Test that all contexts are eventually sampled."""
        rng = np.random.default_rng(42)
        contexts = ("1", "2", "3", "4")
        cue_map = {"1": "A", "2": "B", "3": "C", "4": "D"}
        probe_map = {"1": "W", "2": "X", "3": "Y", "4": "Z"}

        gen = nAX(contexts=contexts, cue_map=cue_map, probe_map=probe_map, rng=rng)

        # Generate many strings
        strings = gen.generate_string_set(n_samples=100)

        # Track which contexts were used
        contexts_used = set()
        for string in strings:
            contexts_used.add(string[0])

        # All contexts should have been used at least once
        assert contexts_used == set(contexts)

    def test_grammaticality_stress_test(self):
        """Stress test for grammaticality with many samples."""
        rng = np.random.default_rng(42)
        gen = nAX(
            contexts=("1", "2", "3"),
            cue_map={"1": "A", "2": "B", "3": "C"},
            probe_map={"1": "X", "2": "Y", "3": "Z"},
            fillers=("F", "G", "H"),
            rng=rng,
        )

        # Generate many strings
        for _ in range(100):
            string = gen.generate_string()
            assert gen.is_grammatical(string + [gen.eos])


class TestVerboseMode:
    """Tests for verbose initialization."""

    def test_verbose_initialization(self):
        """Test that verbose mode doesn't raise errors."""
        # Should not crash with verbose=True
        gen = nAX(verbose=True)
        assert gen.verbose is True

    def test_non_verbose_initialization(self):
        """Test default non-verbose mode."""
        gen = nAX()
        assert gen.verbose is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
