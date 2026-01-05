"""Tests for string-level metrics."""

import pytest
import numpy as np

from symseq.metrics.string import (
    entropy,
    block_entropy,
    entropy_rate,
    emc,
    compressibility,
    lzw_complexity,
    linguistic_complexity,
    permutation_entropy,
)


class TestEntropy:
    """Tests for entropy functions."""

    def test_uniform_distribution(self):
        seq = ['A', 'B', 'C', 'D']
        H = entropy(seq)
        assert H == pytest.approx(2.0, abs=0.01)

    def test_constant_sequence(self):
        seq = ['A'] * 10
        H = entropy(seq)
        assert H == pytest.approx(0.0, abs=0.01)

    def test_binary_sequence(self):
        seq = ['A', 'B']
        H = entropy(seq)
        assert H == pytest.approx(1.0, abs=0.01)

    def test_empty_sequence(self):
        seq = []
        H = entropy(seq)
        assert H == 0.0


class TestBlockEntropy:
    """Tests for block_entropy function."""

    def test_block_size_1(self):
        seq = ['A', 'B', 'A', 'B']
        H = block_entropy(seq, block_size=1)
        H_regular = entropy(seq)
        assert H == pytest.approx(H_regular, abs=0.01)

    def test_block_size_2(self):
        seq = ['A', 'B'] * 10
        H = block_entropy(seq, block_size=2)
        assert H > 0

    def test_sequence_too_short(self):
        seq = ['A', 'B']
        H = block_entropy(seq, block_size=5)
        assert H == 0.0


class TestEntropyRate:
    """Tests for entropy_rate function."""

    def test_iid_sequence(self):
        np.random.seed(42)
        seq = list(np.random.choice(['A', 'B'], size=100))
        h_mu = entropy_rate(seq, max_block_size=3)
        H = entropy(seq)
        assert h_mu <= H

    def test_deterministic_sequence(self):
        seq = ['A', 'B'] * 50
        h_mu = entropy_rate(seq, max_block_size=3)
        assert h_mu >= 0


class TestEMC:
    """Tests for Effective Measure Complexity."""

    def test_iid_sequence(self):
        np.random.seed(42)
        seq = list(np.random.choice(['A', 'B'], size=100))
        emc_val, h_mu = emc(seq, max_block_size=5)
        assert emc_val >= 0
        assert h_mu >= 0

    def test_deterministic_sequence(self):
        seq = ['A'] * 50
        emc_val, h_mu = emc(seq, max_block_size=5)
        assert emc_val >= 0


class TestCompressibility:
    """Tests for compressibility function."""

    def test_repetitive_sequence(self):
        seq = ['A'] * 100
        C = compressibility(seq)
        assert 0 < C < 0.5

    def test_random_sequence(self):
        np.random.seed(42)
        seq = list(np.random.choice(['A', 'B', 'C', 'D'], size=100))
        C = compressibility(seq)
        assert 0 < C <= 1.0

    def test_range(self):
        seq = ['A', 'B', 'C'] * 10
        C = compressibility(seq)
        assert 0 < C <= 1.0


class TestLZWComplexity:
    """Tests for LZW complexity."""

    def test_normalized(self):
        seq = ['A', 'B', 'A', 'B'] * 10
        C = lzw_complexity(seq, normalized=True)
        assert 0 < C <= 1.0

    def test_unnormalized(self):
        seq = ['A', 'B', 'A', 'B'] * 10
        C = lzw_complexity(seq, normalized=False)
        assert C > 0
        assert isinstance(C, (int, float))


class TestLinguisticComplexity:
    """Tests for linguistic_complexity function."""

    def test_uniform_sequence(self):
        seq = ['A', 'B', 'C', 'D']
        C = linguistic_complexity(seq)
        assert C == pytest.approx(1.0, abs=0.01)

    def test_repetitive_sequence(self):
        seq = ['A'] * 10
        C = linguistic_complexity(seq)
        assert C == pytest.approx(1.0, abs=0.01)

    def test_empty_sequence(self):
        seq = []
        C = linguistic_complexity(seq)
        assert C == 0.0

    def test_range(self):
        seq = ['A', 'B', 'C'] * 5
        C = linguistic_complexity(seq)
        assert 0 <= C <= 1.0


class TestPermutationEntropy:
    """Tests for permutation_entropy function."""

    def test_constant_sequence(self):
        seq = [5.0] * 100
        H = permutation_entropy(seq, order=3, normalize=True)
        assert H <= 0.01

    def test_random_sequence(self):
        np.random.seed(42)
        seq = np.random.randn(1000)
        H = permutation_entropy(seq, order=3, normalize=True)
        assert 0.9 < H <= 1.0

    def test_symbolic_sequence(self):
        seq = ['A', 'B', 'C'] * 30
        H = permutation_entropy(seq, order=3, normalize=True)
        assert 0 <= H <= 1.0

    def test_range(self):
        seq = np.random.randn(100)
        H = permutation_entropy(seq, order=3, normalize=True)
        assert 0 <= H <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
