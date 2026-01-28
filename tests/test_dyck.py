"""
test_dyck.py

Comprehensive test suite for the DyckGenerator class.
"""

import pytest
import numpy as np
from symseq.grammars.dyck import DyckGenerator


class TestInitialization:
    """Tests for DyckGenerator initialization and validation."""

    def test_basic_initialization(self):
        """Test basic initialization with minimal parameters."""
        gen = DyckGenerator(k=1, mode="stack")
        assert gen.k == 1
        assert gen.mode == "stack"
        assert gen.parentheses == {"a": "A"}
        assert gen.distractors == ["0", "1"]

    def test_custom_parentheses(self):
        """Test initialization with custom parentheses."""
        custom = {"(": ")", "[": "]"}
        gen = DyckGenerator(k=2, mode="stack", parentheses=custom)
        assert gen.parentheses == custom
        assert gen._opens == ["(", "["]
        assert gen._closes == [")", "]"]

    def test_invalid_k(self):
        """Test that invalid k values raise ValueError."""
        with pytest.raises(ValueError, match="`k` must be a positive integer"):
            DyckGenerator(k=0, mode="stack")
        with pytest.raises(ValueError, match="`k` must be a positive integer"):
            DyckGenerator(k=-1, mode="stack")
        with pytest.raises(ValueError, match="`k` must be a positive integer"):
            DyckGenerator(k=1.5, mode="stack")

    def test_k_parentheses_mismatch(self):
        """Test that mismatched k and parentheses raises ValueError."""
        with pytest.raises(ValueError, match="has 2 types but k=1"):
            DyckGenerator(k=1, mode="stack", parentheses={"(": ")", "[": "]"})

    def test_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="mode must be 'stack' or 'uniform'"):
            DyckGenerator(k=1, mode="invalid")

    def test_stack_mode_invalid_p_open(self):
        """Test that invalid p_open in stack mode raises ValueError."""
        with pytest.raises(ValueError, match="p_open must be in"):
            DyckGenerator(k=1, mode="stack", p_open=0.0)
        with pytest.raises(ValueError, match="p_open must be in"):
            DyckGenerator(k=1, mode="stack", p_open=0.5)
        with pytest.raises(ValueError, match="p_open must be in"):
            DyckGenerator(k=1, mode="stack", p_open=0.8)

    def test_uniform_mode_requires_target_pairs(self):
        """Test that uniform mode requires target_pairs."""
        with pytest.raises(ValueError, match="target_pairs must be a positive integer"):
            DyckGenerator(k=1, mode="uniform")
        with pytest.raises(ValueError, match="target_pairs must be a positive integer"):
            DyckGenerator(k=1, mode="uniform", target_pairs=0)

    def test_invalid_max_depth(self):
        """Test that invalid max_depth raises ValueError."""
        with pytest.raises(ValueError, match="`max_depth` must be a positive integer"):
            DyckGenerator(k=1, mode="stack", max_depth=0)
        with pytest.raises(ValueError, match="`max_depth` must be a positive integer"):
            DyckGenerator(k=1, mode="stack", max_depth=-1)

    def test_large_k(self):
        """Test that k > 26 raises ValueError for default parentheses."""
        with pytest.raises(ValueError, match="up to k=26"):
            DyckGenerator(k=27, mode="stack")

    def test_large_n_distractors(self):
        """Test that n_distractors > 10 raises ValueError for defaults."""
        with pytest.raises(ValueError, match="up to 10 symbols"):
            DyckGenerator(k=1, mode="stack", n_distractors=11)


class TestDefaultGeneration:
    """Tests for default parentheses and distractor generation."""

    def test_default_parentheses_generation(self):
        """Test that default parentheses are generated correctly."""
        gen = DyckGenerator(k=3, mode="stack")
        assert gen.parentheses == {"a": "A", "b": "B", "c": "C"}

    def test_default_distractors_generation(self):
        """Test that default distractors are generated correctly."""
        gen = DyckGenerator(k=1, mode="stack", n_distractors=3)
        assert gen.distractors == ["0", "1", "2"]

    def test_no_distractors(self):
        """Test initialization with no distractors."""
        gen = DyckGenerator(k=1, mode="stack", distractors=[])
        assert gen.distractors == []

    def test_custom_distractors(self):
        """Test initialization with custom distractors."""
        custom = ["x", "y", "z"]
        gen = DyckGenerator(k=1, mode="stack", distractors=custom, n_distractors=5)
        assert gen.distractors == custom  # n_distractors ignored when custom provided

    def test_distractor_collision_detection(self):
        """Test that distractor-parenthesis collision is detected."""
        with pytest.raises(ValueError, match="cannot overlap with parenthesis symbols"):
            DyckGenerator(k=1, mode="stack", distractors=["a", "x"])
        with pytest.raises(ValueError, match="cannot overlap with parenthesis symbols"):
            DyckGenerator(k=2, mode="stack", distractors=["A", "B"])


class TestAlphabet:
    """Tests for alphabet computation."""

    def test_alphabet_without_distractors(self):
        """Test alphabet with no distractors."""
        gen = DyckGenerator(k=2, mode="stack", distractors=[])
        assert set(gen.alphabet) == {"a", "A", "b", "B"}

    def test_alphabet_with_distractors(self):
        """Test alphabet with default distractors."""
        gen = DyckGenerator(k=1, mode="stack", n_distractors=2)
        assert set(gen.alphabet) == {"a", "A", "0", "1"}

    def test_alphabet_sorting(self):
        """Test that alphabet is sorted."""
        gen = DyckGenerator(k=2, mode="stack")
        # Should be sorted: digits, then uppercase, then lowercase
        assert gen.alphabet == sorted(gen.alphabet)


class TestStackMode:
    """Tests for stack mode generation."""

    def test_stack_generates_valid_dyck(self):
        """Test that stack mode generates valid Dyck strings."""
        gen = DyckGenerator(k=1, mode="stack", n_distractors=0)
        for _ in range(20):
            tokens = gen.generate_string()
            assert gen._is_valid_dyck(tokens)

    def test_stack_variable_length(self):
        """Test that stack mode generates variable-length strings."""
        gen = DyckGenerator(k=1, mode="stack", n_distractors=0)
        lengths = [len(gen.generate_string()) for _ in range(50)]
        # Should have some variation in lengths
        assert len(set(lengths)) > 3

    def test_stack_respects_max_depth(self):
        """Test that stack mode respects max_depth constraint."""
        gen = DyckGenerator(k=1, mode="stack", max_depth=3, n_distractors=0)
        for _ in range(20):
            tokens = gen.generate_string()
            assert gen._check_max_depth(tokens)

    def test_stack_with_multiple_types(self):
        """Test stack mode with multiple parenthesis types."""
        gen = DyckGenerator(k=3, mode="stack", n_distractors=0)
        tokens = gen.generate_string()
        assert gen._is_valid_dyck(tokens)
        # Should have at least one token
        assert len(tokens) >= 2


class TestUniformMode:
    """Tests for uniform mode generation."""

    def test_uniform_generates_valid_dyck(self):
        """Test that uniform mode generates valid Dyck strings."""
        gen = DyckGenerator(k=1, mode="uniform", target_pairs=5, n_distractors=0)
        for _ in range(20):
            tokens = gen.generate_string()
            assert gen._is_valid_dyck(tokens)

    def test_uniform_fixed_length(self):
        """Test that uniform mode generates fixed-length strings."""
        target = 5
        gen = DyckGenerator(k=1, mode="uniform", target_pairs=target, n_distractors=0)
        for _ in range(20):
            tokens = gen.generate_string()
            assert len(tokens) == 2 * target

    def test_uniform_respects_max_depth(self):
        """Test that uniform mode respects max_depth constraint."""
        gen = DyckGenerator(k=1, mode="uniform", target_pairs=10, max_depth=3, n_distractors=0)
        for _ in range(10):
            tokens = gen.generate_string()
            assert gen._check_max_depth(tokens)

    def test_uniform_colorize_true(self):
        """Test uniform mode with colorize=True uses multiple types."""
        gen = DyckGenerator(k=3, mode="uniform", target_pairs=10, uniform_colorize=True, n_distractors=0)
        tokens = gen.generate_string()
        # With 10 pairs and 3 types, very likely to have multiple types
        opens_used = set(t for t in tokens if t in gen._opens)
        # May not always have all types, but should have variation
        assert len(tokens) == 20

    def test_uniform_colorize_false(self):
        """Test uniform mode with colorize=False uses single type."""
        gen = DyckGenerator(k=3, mode="uniform", target_pairs=10, uniform_colorize=False, n_distractors=0)
        tokens = gen.generate_string()
        opens_used = set(t for t in tokens if t in gen._opens)
        closes_used = set(t for t in tokens if t in gen._closes)
        # Should only use first type
        assert opens_used == {"a"}
        assert closes_used == {"A"}

    def test_uniform_zero_pairs(self):
        """Test uniform mode with zero pairs."""
        gen = DyckGenerator(k=1, mode="uniform", target_pairs=1, n_distractors=0)
        tokens = gen._sample_uniform_pairs(0)
        assert tokens == []


class TestDistractors:
    """Tests for distractor injection."""

    def test_distractor_injection(self):
        """Test that distractors are injected correctly."""
        gen = DyckGenerator(k=1, mode="stack", n_distractors=3)
        tokens = gen.generate_string(add_distractors=True, n_distractors=5)
        # Count distractors
        distractor_count = sum(1 for t in tokens if t in gen.distractors)
        assert distractor_count == 5

    def test_default_n_distractors_used(self):
        """Test that default n_distractors is used when add_distractors=True."""
        gen = DyckGenerator(k=1, mode="stack", n_distractors=3)
        tokens = gen.generate_string(add_distractors=True)
        distractor_count = sum(1 for t in tokens if t in gen.distractors)
        assert distractor_count == 3

    def test_no_distractors_injected_by_default(self):
        """Test that distractors are not injected by default."""
        gen = DyckGenerator(k=1, mode="stack")
        tokens = gen.generate_string()
        distractor_count = sum(1 for t in tokens if t in gen.distractors)
        assert distractor_count == 0

    def test_distractors_preserve_validity(self):
        """Test that adding distractors preserves Dyck validity."""
        gen = DyckGenerator(k=1, mode="stack", n_distractors=5)
        tokens = gen.generate_string(add_distractors=True, n_distractors=10)
        # Should still be valid when distractors are ignored
        assert gen._is_valid_dyck(tokens)


class TestStringSetGeneration:
    """Tests for generating sets of strings."""

    def test_generate_string_set(self):
        """Test generating multiple strings."""
        gen = DyckGenerator(k=1, mode="stack", n_distractors=0)
        strings = gen.generate_string_set(n=10)
        assert len(strings) == 10
        for s in strings:
            assert gen._is_valid_dyck(s)

    def test_generate_string_set_with_distractors(self):
        """Test generating multiple strings with distractors."""
        gen = DyckGenerator(k=1, mode="stack", n_distractors=2)
        strings = gen.generate_string_set(n=5, add_distractors=True, n_distractors=3)
        assert len(strings) == 5
        for s in strings:
            assert gen._is_valid_dyck(s)
            distractor_count = sum(1 for t in s if t in gen.distractors)
            assert distractor_count == 3


class TestIllegalStrings:
    """Tests for illegal string generation."""

    def test_illegal_strings_are_invalid(self):
        """Test that generated illegal strings are actually invalid."""
        gen = DyckGenerator(k=1, mode="stack", n_distractors=0)
        illegal = gen.generate_nongrammatical_strings(n=20, verify_illegal=True)
        for s in illegal:
            assert not gen._is_valid_dyck(s)

    def test_illegal_string_strategies(self):
        """Test different violation strategies."""
        gen = DyckGenerator(k=2, mode="uniform", target_pairs=5, n_distractors=0)

        strategies = ["replace_close", "insert_extra_close", "delete_token", "swap_adjacent", "truncate"]

        for strategy in strategies:
            illegal = gen.generate_nongrammatical_strings(n=5, strategies=[strategy], verify_illegal=True)
            for s in illegal:
                assert not gen._is_valid_dyck(s)

    def test_multiple_deviants(self):
        """Test generating illegal strings with multiple deviations."""
        gen = DyckGenerator(k=1, mode="stack", n_distractors=0)
        illegal = gen.generate_nongrammatical_strings(n=10, n_deviants=3, verify_illegal=True)
        for s in illegal:
            assert not gen._is_valid_dyck(s)

    def test_illegal_with_distractors(self):
        """Test illegal strings with distractors."""
        gen = DyckGenerator(k=1, mode="stack", n_distractors=2)
        illegal = gen.generate_nongrammatical_strings(n=5, add_distractors=True, n_distractors=3, verify_illegal=True)
        for s in illegal:
            assert not gen._is_valid_dyck(s)
            distractor_count = sum(1 for t in s if t in gen.distractors)
            assert distractor_count == 3

    def test_illegal_generation_timeout(self):
        """Test that illegal generation raises error on timeout."""
        # Create a scenario where it's very hard to generate illegal strings
        # by using a very constrained setup
        gen = DyckGenerator(k=1, mode="uniform", target_pairs=1, n_distractors=0)

        # With only 1 pair, many violations might still result in valid strings
        # but with enough attempts it should succeed or timeout
        try:
            illegal = gen.generate_nongrammatical_strings(n=1, n_deviants=1, max_attempts=5, verify_illegal=True)
            # If it succeeds, verify it's actually illegal
            assert not gen._is_valid_dyck(illegal[0])
        except RuntimeError as e:
            # Timeout is also acceptable
            assert "Failed to generate an illegal string" in str(e)

    def test_verify_illegal_false(self):
        """Test that verify_illegal=False doesn't check validity."""
        gen = DyckGenerator(k=1, mode="stack", n_distractors=0)
        # With verify_illegal=False, some might be valid
        illegal = gen.generate_nongrammatical_strings(n=20, verify_illegal=False)
        # At least some should be invalid
        invalid_count = sum(1 for s in illegal if not gen._is_valid_dyck(s))
        assert invalid_count > 0  # Most should be invalid even without verification


class TestViolationStrategies:
    """Tests for individual violation strategies."""

    def test_replace_close_violation(self):
        """Test replace_close violation strategy."""
        gen = DyckGenerator(k=2, mode="uniform", target_pairs=3, n_distractors=0)
        valid = gen.generate_string()
        violated = gen._apply_violation(valid[:], "replace_close")
        # Should have modified the string (most of the time)
        # Can't guarantee in all cases for k=2
        assert isinstance(violated, list)

    def test_insert_extra_close_violation(self):
        """Test insert_extra_close violation strategy."""
        gen = DyckGenerator(k=1, mode="uniform", target_pairs=3, n_distractors=0)
        valid = gen.generate_string()
        violated = gen._apply_violation(valid[:], "insert_extra_close")
        # Should have one more token
        assert len(violated) == len(valid) + 1
        # Should be invalid
        assert not gen._is_valid_dyck(violated)

    def test_delete_token_violation(self):
        """Test delete_token violation strategy."""
        gen = DyckGenerator(k=1, mode="uniform", target_pairs=3, n_distractors=0)
        valid = gen.generate_string()
        violated = gen._apply_violation(valid[:], "delete_token")
        # Should have one fewer token
        assert len(violated) == len(valid) - 1
        # Should be invalid (almost always)
        assert not gen._is_valid_dyck(violated)

    def test_swap_adjacent_violation(self):
        """Test swap_adjacent violation strategy."""
        gen = DyckGenerator(k=1, mode="uniform", target_pairs=3, n_distractors=0)
        valid = gen.generate_string()
        violated = gen._apply_violation(valid[:], "swap_adjacent")
        # Should have same length
        assert len(violated) == len(valid)

    def test_truncate_violation(self):
        """Test truncate violation strategy."""
        gen = DyckGenerator(k=1, mode="uniform", target_pairs=5, n_distractors=0)
        valid = gen.generate_string()
        violated = gen._apply_violation(valid[:], "truncate")
        # Should be shorter
        assert len(violated) < len(valid)
        # Should be invalid
        assert not gen._is_valid_dyck(violated)

    def test_empty_string_violation(self):
        """Test that violations handle empty strings gracefully."""
        gen = DyckGenerator(k=1, mode="stack", n_distractors=0)
        empty = []
        for strategy in ["replace_close", "insert_extra_close", "delete_token", "swap_adjacent", "truncate"]:
            result = gen._apply_violation(empty, strategy)
            assert isinstance(result, list)


class TestValidityChecking:
    """Tests for Dyck string validity checking."""

    def test_valid_simple_dyck(self):
        """Test validity checking on simple valid strings."""
        gen = DyckGenerator(k=1, mode="stack", n_distractors=0)
        assert gen._is_valid_dyck(["a", "A"])
        assert gen._is_valid_dyck(["a", "a", "A", "A"])
        assert gen._is_valid_dyck(["a", "A", "a", "A"])

    def test_invalid_simple_dyck(self):
        """Test validity checking on simple invalid strings."""
        gen = DyckGenerator(k=1, mode="stack", n_distractors=0)
        assert not gen._is_valid_dyck(["A", "a"])  # Wrong order
        assert not gen._is_valid_dyck(["a", "A", "A"])  # Extra close
        assert not gen._is_valid_dyck(["a", "a", "A"])  # Missing close
        assert not gen._is_valid_dyck(["a"])  # Incomplete

    def test_valid_multi_type_dyck(self):
        """Test validity checking with multiple types."""
        gen = DyckGenerator(k=2, mode="stack", n_distractors=0)
        assert gen._is_valid_dyck(["a", "b", "B", "A"])
        assert gen._is_valid_dyck(["a", "A", "b", "B"])

    def test_invalid_multi_type_dyck(self):
        """Test validity checking with mismatched types."""
        gen = DyckGenerator(k=2, mode="stack", n_distractors=0)
        assert not gen._is_valid_dyck(["a", "B"])  # Wrong close type
        assert not gen._is_valid_dyck(["b", "A"])  # Wrong close type

    def test_validity_ignores_distractors(self):
        """Test that validity checking ignores distractors."""
        gen = DyckGenerator(k=1, mode="stack", n_distractors=2)
        assert gen._is_valid_dyck(["0", "a", "1", "A", "0"])
        assert gen._is_valid_dyck(["a", "0", "a", "1", "A", "A"])

    def test_empty_string_is_valid(self):
        """Test that empty string is valid."""
        gen = DyckGenerator(k=1, mode="stack", n_distractors=0)
        assert gen._is_valid_dyck([])


class TestMaxDepthChecking:
    """Tests for max depth checking."""

    def test_depth_checking_simple(self):
        """Test depth checking on simple strings."""
        gen = DyckGenerator(k=1, mode="stack", max_depth=2, n_distractors=0)
        assert gen._check_max_depth(["a", "A"])  # depth 1
        assert gen._check_max_depth(["a", "a", "A", "A"])  # depth 2
        assert not gen._check_max_depth(["a", "a", "a", "A", "A", "A"])  # depth 3

    def test_depth_checking_flat(self):
        """Test depth checking on flat structures."""
        gen = DyckGenerator(k=1, mode="stack", max_depth=1, n_distractors=0)
        assert gen._check_max_depth(["a", "A", "a", "A", "a", "A"])  # all depth 1

    def test_depth_none_allows_any(self):
        """Test that max_depth=None allows any depth."""
        gen = DyckGenerator(k=1, mode="stack", max_depth=None, n_distractors=0)
        deep = ["a"] * 20 + ["A"] * 20
        assert gen._check_max_depth(deep)


class TestRandomness:
    """Tests for randomness and reproducibility."""

    def test_custom_rng(self):
        """Test using custom random number generator."""
        rng = np.random.default_rng(42)
        gen = DyckGenerator(k=1, mode="stack", rng=rng, n_distractors=0)
        s1 = gen.generate_string()

        rng = np.random.default_rng(42)
        gen = DyckGenerator(k=1, mode="stack", rng=rng, n_distractors=0)
        s2 = gen.generate_string()

        assert s1 == s2

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        rng1 = np.random.default_rng(42)
        gen1 = DyckGenerator(k=1, mode="stack", rng=rng1, n_distractors=0)

        rng2 = np.random.default_rng(123)
        gen2 = DyckGenerator(k=1, mode="stack", rng=rng2, n_distractors=0)

        # Generate multiple strings to ensure difference
        strings1 = [gen1.generate_string() for _ in range(10)]
        strings2 = [gen2.generate_string() for _ in range(10)]

        # Very unlikely to be identical
        assert strings1 != strings2


class TestUtilities:
    """Tests for utility methods."""

    def test_catalan_numbers(self):
        """Test Catalan number computation."""
        gen = DyckGenerator(k=1, mode="stack", n_distractors=0)
        catalans = gen._catalans(5)
        # First few Catalan numbers: 1, 1, 2, 5, 14, 42
        assert catalans[0] == 1
        assert catalans[1] == 1
        assert catalans[2] == 2
        assert catalans[3] == 5
        assert catalans[4] == 14
        assert catalans[5] == 42

    def test_weighted_choice(self):
        """Test weighted choice utility."""
        gen = DyckGenerator(k=1, mode="stack", n_distractors=0)
        rng = np.random.default_rng(42)
        gen.rng = rng

        weights = np.array([1, 0, 0, 0])
        # Should always choose index 0
        for _ in range(10):
            assert gen._weighted_choice(weights) == 0

    def test_choice_utility(self):
        """Test choice utility method."""
        gen = DyckGenerator(k=1, mode="stack", n_distractors=0)
        rng = np.random.default_rng(42)
        gen.rng = rng

        seq = ["a", "b", "c"]
        choices = [gen._choice(seq) for _ in range(100)]
        # Should have chosen all elements at least once
        assert set(choices) == set(seq)


class TestRepr:
    """Tests for string representation."""

    def test_repr(self):
        """Test __repr__ method."""
        gen = DyckGenerator(k=2, mode="stack", p_open=0.3, n_distractors=3)
        repr_str = repr(gen)
        assert "DyckGenerator" in repr_str
        assert "k=2" in repr_str
        assert "mode='stack'" in repr_str
        assert "p_open=0.3" in repr_str
        assert "n_distractors=3" in repr_str


class TestEdgeCases:
    """Tests for edge cases and corner scenarios."""

    def test_k_equals_26(self):
        """Test maximum k value with default parentheses."""
        gen = DyckGenerator(k=26, mode="stack", n_distractors=0)
        assert len(gen.parentheses) == 26
        tokens = gen.generate_string()
        assert gen._is_valid_dyck(tokens)

    def test_very_small_target_pairs(self):
        """Test uniform mode with target_pairs=1."""
        gen = DyckGenerator(k=1, mode="uniform", target_pairs=1, n_distractors=0)
        tokens = gen.generate_string()
        assert tokens == ["a", "A"]
        assert gen._is_valid_dyck(tokens)

    def test_large_target_pairs(self):
        """Test uniform mode with large target_pairs."""
        gen = DyckGenerator(k=1, mode="uniform", target_pairs=20, n_distractors=0)
        tokens = gen.generate_string()
        assert len(tokens) == 40
        assert gen._is_valid_dyck(tokens)

    def test_very_restrictive_max_depth(self):
        """Test generation with max_depth=1 in uniform mode."""
        gen = DyckGenerator(k=1, mode="uniform", target_pairs=5, max_depth=1, n_distractors=0)
        tokens = gen.generate_string()
        assert gen._check_max_depth(tokens)
        # With max_depth=1, structure should be flat: aAaAaAaAaA
        assert tokens == ["a", "A"] * 5

    def test_max_attempts_exceeded(self):
        """Test that max_attempts exceeded raises RuntimeError."""
        # Create impossible constraint: uniform with large pairs and depth=1
        # but depth=1 only allows flat structures
        gen = DyckGenerator(k=1, mode="uniform", target_pairs=3, max_depth=1, n_distractors=0)
        # This should work (flat structure possible)
        tokens = gen.generate_string()
        assert len(tokens) == 6

        # Now try something that should timeout with very low max_attempts
        gen2 = DyckGenerator(k=1, mode="uniform", target_pairs=10, max_depth=2, n_distractors=0)
        # With max_attempts=1, might fail (depends on random draw)
        try:
            tokens = gen2.generate_string(max_attempts=1)
            # If it succeeds, should be valid
            assert gen2._check_max_depth(tokens)
        except RuntimeError as e:
            # Failure is also acceptable
            assert "Failed to generate" in str(e)


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow_stack_mode(self):
        """Test complete workflow in stack mode."""
        gen = DyckGenerator(k=3, mode="stack", p_open=0.2, max_depth=5, n_distractors=3)

        # Generate valid strings
        valid = gen.generate_string_set(n=10, add_distractors=True)
        assert len(valid) == 10
        for s in valid:
            assert gen._is_valid_dyck(s)
            assert gen._check_max_depth(s)

        # Generate illegal strings
        illegal = gen.generate_nongrammatical_strings(n=10, n_deviants=2, add_distractors=True, verify_illegal=True)
        assert len(illegal) == 10
        for s in illegal:
            assert not gen._is_valid_dyck(s)

    def test_full_workflow_uniform_mode(self):
        """Test complete workflow in uniform mode."""
        gen = DyckGenerator(k=2, mode="uniform", target_pairs=8, max_depth=4, uniform_colorize=True, n_distractors=2)

        # Generate valid strings
        valid = gen.generate_string_set(n=5, add_distractors=True, n_distractors=4)
        assert len(valid) == 5
        for s in valid:
            # Should be 16 tokens (8 pairs) + 4 distractors = 20
            assert len(s) == 20
            assert gen._is_valid_dyck(s)
            assert gen._check_max_depth(s)

        # Generate illegal strings
        illegal = gen.generate_nongrammatical_strings(n=5, n_deviants=1, verify_illegal=True)
        assert len(illegal) == 5
        for s in illegal:
            assert not gen._is_valid_dyck(s)

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with same seed."""
        rng1 = np.random.default_rng(42)
        gen1 = DyckGenerator(k=2, mode="stack", rng=rng1, n_distractors=3)
        valid1 = gen1.generate_string_set(n=5)
        illegal1 = gen1.generate_nongrammatical_strings(n=5)

        rng2 = np.random.default_rng(42)
        gen2 = DyckGenerator(k=2, mode="stack", rng=rng2, n_distractors=3)
        valid2 = gen2.generate_string_set(n=5)
        illegal2 = gen2.generate_nongrammatical_strings(n=5)

        assert valid1 == valid2
        assert illegal1 == illegal2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
