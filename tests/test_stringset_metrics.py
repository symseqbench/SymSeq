"""Tests for string-set level metrics."""

import pytest
import numpy as np

from symseq.metrics.stringset import (
    hamming_distance,
    edit_distance,
    normalized_compression_distance,
    pairwise_distances,
    mutual_information_strings,
    normalized_mutual_information,
    string_set_entropy,
    global_acs_knowlton96,
    anchor_acs_knowlton96,
    acs_bailey2008,
)


class TestDistances:
    """Tests for distance metrics."""

    def test_hamming_distance_identical(self):
        seq1 = ['A', 'B', 'C']
        seq2 = ['A', 'B', 'C']
        d = hamming_distance(seq1, seq2)
        assert d == 0

    def test_hamming_distance_different(self):
        seq1 = ['A', 'B', 'C']
        seq2 = ['A', 'X', 'C']
        d = hamming_distance(seq1, seq2)
        assert d == 1

    def test_hamming_distance_length_mismatch(self):
        seq1 = ['A', 'B']
        seq2 = ['A', 'B', 'C']
        with pytest.raises(ValueError):
            hamming_distance(seq1, seq2)

    def test_edit_distance_identical(self):
        seq1 = ['A', 'B', 'C']
        seq2 = ['A', 'B', 'C']
        d = edit_distance(seq1, seq2)
        assert d == 0

    def test_edit_distance_insertion(self):
        seq1 = ['A', 'B']
        seq2 = ['A', 'X', 'B']
        d = edit_distance(seq1, seq2)
        assert d == 1

    def test_ncd_identical(self):
        seq1 = ['A', 'B', 'C'] * 10
        seq2 = ['A', 'B', 'C'] * 10
        ncd = normalized_compression_distance(seq1, seq2)
        assert ncd < 0.2

    def test_ncd_different(self):
        seq1 = ['A', 'B', 'C'] * 10
        seq2 = ['X', 'Y', 'Z'] * 10
        ncd = normalized_compression_distance(seq1, seq2)
        assert ncd > 0.1


class TestPairwiseDistances:
    """Tests for pairwise_distances function."""

    def test_edit_metric(self):
        strings = [['A', 'B'], ['A', 'C'], ['B', 'C']]
        D = pairwise_distances(strings, metric='edit')
        assert D.shape == (3, 3)
        assert D[0, 0] == 0
        assert D[0, 1] == D[1, 0]

    def test_hamming_metric(self):
        strings = [['A', 'B'], ['A', 'C'], ['B', 'C']]
        D = pairwise_distances(strings, metric='hamming')
        assert D.shape == (3, 3)
        assert D[0, 0] == 0

    def test_ncd_metric(self):
        strings = [['A', 'B'] * 5, ['A', 'C'] * 5, ['B', 'C'] * 5]
        D = pairwise_distances(strings, metric='ncd')
        assert D.shape == (3, 3)
        assert D[0, 0] == 0


class TestInformationMetrics:
    """Tests for information-theoretic metrics."""

    def test_mi_identical_sequences(self):
        seq = ['A', 'B', 'C', 'A']
        MI = mutual_information_strings(seq, seq, method='joint')
        assert MI > 0

    def test_mi_independent_sequences(self):
        np.random.seed(42)
        seq1 = list(np.random.choice(['A', 'B'], size=100))
        seq2 = list(np.random.choice(['A', 'B'], size=100))
        MI = mutual_information_strings(seq1, seq2, method='joint')
        assert MI >= 0

    def test_nmi_range(self):
        seq1 = ['A', 'B', 'C']
        seq2 = ['A', 'B', 'C']
        NMI = normalized_mutual_information(seq1, seq2)
        assert 0 <= NMI <= 1.0

    def test_nmi_identical(self):
        seq = ['A', 'B', 'C', 'A']
        NMI = normalized_mutual_information(seq, seq)
        assert NMI == pytest.approx(1.0, abs=0.01)

    def test_string_set_entropy_uniform(self):
        strings = [['A', 'B'], ['C', 'D'], ['E', 'F'], ['G', 'H']] * 10
        H = string_set_entropy(strings, use_frequencies=False)
        assert H == pytest.approx(2.0, abs=0.01)

    def test_string_set_entropy_with_frequencies(self):
        strings = [['A', 'B']] * 50 + [['C', 'D']] * 50
        H = string_set_entropy(strings, use_frequencies=True)
        assert H == pytest.approx(1.0, abs=0.01)


class TestChunkStrength:
    """Tests for ACS metrics."""

    def test_acs_bailey2008(self):
        train = [['A', 'B'], ['A', 'B'], ['C', 'D']]
        test = [['A', 'B'], ['E', 'F']]
        strengths = acs_bailey2008(train, test, n_range=(2, 2))
        assert len(strengths) == 2
        assert strengths[0] > strengths[1]

    def test_global_acs(self):
        train = [['A', 'B', 'C']] * 10
        test = [['A', 'B', 'C'], ['X', 'Y', 'Z']]
        strengths = global_acs_knowlton96(train, test, n_range=(2, 2))
        assert len(strengths) == 2
        assert strengths[0] > strengths[1]

    def test_anchor_acs(self):
        train = [['A', 'B', 'C']] * 10
        test = [['A', 'B', 'C'], ['X', 'Y', 'Z']]
        strengths = anchor_acs_knowlton96(train, test, n_range=(2, 2))
        assert len(strengths) == 2
        assert all(s >= 0 for s in strengths)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
