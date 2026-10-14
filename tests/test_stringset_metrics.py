"""Tests for string-set level metrics."""

import warnings

import numpy as np
import pytest

from symseq.metrics import average_string_similarity, string_similarity
from symseq.metrics.stringset import (
    acs_bailey2008,
    anchor_acs_knowlton96,
    edit_distance,
    global_acs_knowlton96,
    hamming_distance,
    mutual_information_strings,
    normalized_compression_distance,
    normalized_mutual_information,
    pairwise_distances,
    pairwise_distances_parallel,
    pairwise_mutual_information,
    pairwise_mutual_information_parallel,
    string_set_entropy,
)


class TestDistances:
    """Tests for distance metrics."""

    def test_hamming_distance_identical(self):
        seq1 = ["A", "B", "C"]
        seq2 = ["A", "B", "C"]
        d = hamming_distance(seq1, seq2)
        assert d == 0

    def test_hamming_distance_different(self):
        seq1 = ["A", "B", "C"]
        seq2 = ["A", "X", "C"]
        d = hamming_distance(seq1, seq2)
        assert d == 1

    def test_hamming_distance_length_mismatch(self):
        seq1 = ["A", "B"]
        seq2 = ["A", "B", "C"]
        with pytest.raises(ValueError):
            hamming_distance(seq1, seq2)

    def test_edit_distance_identical(self):
        seq1 = ["A", "B", "C"]
        seq2 = ["A", "B", "C"]
        d = edit_distance(seq1, seq2)
        assert d == 0

    def test_edit_distance_insertion(self):
        seq1 = ["A", "B"]
        seq2 = ["A", "X", "B"]
        d = edit_distance(seq1, seq2)
        assert d == 1

    def test_ncd_identical(self):
        seq1 = ["A", "B", "C"] * 10
        seq2 = ["A", "B", "C"] * 10
        ncd = normalized_compression_distance(seq1, seq2)
        assert ncd < 0.25
        assert ncd < normalized_compression_distance(seq1, ["X", "Y", "Z"] * 10)

    def test_ncd_different(self):
        seq1 = ["A", "B", "C"] * 10
        seq2 = ["X", "Y", "Z"] * 10
        ncd = normalized_compression_distance(seq1, seq2)
        assert ncd > 0.1

    def test_ncd_short_different_strings_use_full_range(self):
        # With gzip framing included, two different 5-symbol strings scored only 0.2.
        assert normalized_compression_distance(list("ABCDE"), list("VWXYZ")) == pytest.approx(1.0)

    def test_ncd_multichar_symbols_do_not_collide(self):
        # both flatten to 'ABC' * 20, but share no symbol: they are as different as two sequences can be
        assert normalized_compression_distance(["A", "BC"] * 20, ["AB", "C"] * 20) > 0.5

    def test_ncd_rejects_string_set(self):
        with pytest.raises(TypeError, match="single sequence"):
            normalized_compression_distance([["A", "B"]] * 3, ["A", "B"] * 3)

    def test_ncd_empty_payload(self):
        assert normalized_compression_distance([], []) == 0.0


class TestStringSimilarity:
    """string_similarity is a similarity: 1 for identical strings, 0 for disjoint ones."""

    def test_identical_is_one(self):
        assert string_similarity(list("ABC"), list("ABC")) == 1.0

    def test_disjoint_is_zero(self):
        assert string_similarity(list("ABC"), list("XYZ")) == 0.0

    def test_partial(self):
        assert string_similarity(list("ABCD"), list("ABXD")) == pytest.approx(0.75)

    def test_empty_pair_is_identical(self):
        assert string_similarity([], []) == 1.0

    def test_average_is_mean_over_references(self):
        train = [list("ABC"), list("XYZ")]
        assert average_string_similarity(train, list("ABC")) == pytest.approx(0.5)

    def test_closer_to_training_set_scores_higher(self):
        train = [list("ABCD")] * 4
        assert average_string_similarity(train, list("ABCD")) > average_string_similarity(train, list("ABXY"))


class TestPairwiseDistances:
    """Tests for pairwise_distances function."""

    def test_edit_metric(self):
        strings = [["A", "B"], ["A", "C"], ["B", "C"]]
        D = pairwise_distances(strings, metric="edit")
        assert D.shape == (3, 3)
        assert D[0, 0] == 0
        assert D[0, 1] == D[1, 0]

    def test_hamming_metric(self):
        strings = [["A", "B"], ["A", "C"], ["B", "C"]]
        D = pairwise_distances(strings, metric="hamming")
        assert D.shape == (3, 3)
        assert D[0, 0] == 0

    def test_ncd_metric(self):
        strings = [["A", "B"] * 5, ["A", "C"] * 5, ["B", "C"] * 5]
        D = pairwise_distances(strings, metric="ncd")
        assert D.shape == (3, 3)
        assert D[0, 0] == 0


class TestPairwiseErrorSemantics:
    """Serial and parallel pairwise helpers must fail the same way."""

    def test_distances_unequal_length_is_nan_in_both(self):
        strings = [["A", "B"], ["A", "B", "C"]]
        serial = pairwise_distances(strings, metric="hamming")
        parallel = pairwise_distances_parallel(strings, metric="hamming", n_jobs=1)
        assert np.isnan(serial[0, 1]) and np.isnan(parallel[0, 1])

    def test_distances_non_value_error_propagates_in_both(self):
        strings = [[1, 2], [1, 3]]  # hamming_distance calls str.__ne__ -> TypeError
        with pytest.raises(TypeError):
            pairwise_distances(strings, metric="hamming")
        with pytest.raises(TypeError):
            pairwise_distances_parallel(strings, metric="hamming", n_jobs=1)

    def test_mi_unequal_length_is_nan_in_both(self):
        strings = [["A", "B"], ["A", "B", "C"]]
        serial = pairwise_mutual_information(strings, method="aligned")
        parallel = pairwise_mutual_information_parallel(strings, method="aligned", n_jobs=1)
        assert np.isnan(serial[0, 1]) and np.isnan(parallel[0, 1])


class TestInformationMetrics:
    """Tests for information-theoretic metrics."""

    def test_mi_identical_sequences(self):
        seq = ["A", "B", "C", "A"]
        MI = mutual_information_strings(seq, seq, method="joint")
        assert MI > 0

    def test_mi_independent_sequences(self):
        np.random.seed(42)
        seq1 = list(np.random.choice(["A", "B"], size=100))
        seq2 = list(np.random.choice(["A", "B"], size=100))
        MI = mutual_information_strings(seq1, seq2, method="joint")
        assert MI >= 0

    def test_nmi_range(self):
        seq1 = ["A", "B", "C"]
        seq2 = ["A", "B", "C"]
        NMI = normalized_mutual_information(seq1, seq2)
        assert 0 <= NMI <= 1.0

    def test_nmi_identical(self):
        seq = ["A", "B", "C", "A"]
        NMI = normalized_mutual_information(seq, seq)
        assert pytest.approx(1.0, abs=0.01) == NMI

    def test_string_set_entropy_uniform(self):
        strings = [["A", "B"], ["C", "D"], ["E", "F"], ["G", "H"]] * 10
        H = string_set_entropy(strings, use_frequencies=False)
        assert pytest.approx(2.0, abs=0.01) == H

    def test_string_set_entropy_with_frequencies(self):
        strings = [["A", "B"]] * 50 + [["C", "D"]] * 50
        H = string_set_entropy(strings, use_frequencies=True)
        assert pytest.approx(1.0, abs=0.01) == H

    def test_string_set_entropy_uniform_is_log2_unique(self):
        strings = [[str(i)] for i in range(32)] + [["0"]] * 50  # repeats are ignored
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            H = string_set_entropy(strings, use_frequencies=False)
        assert pytest.approx(5.0, abs=1e-12) == H

    def test_string_set_entropy_warns_on_few_distinct_strings(self):
        strings = [["A"], ["B"], ["C"]] * 20  # 60 strings, only 3 distinct
        with pytest.warns(UserWarning, match="too small"):
            string_set_entropy(strings, use_frequencies=False)


class TestChunkStrength:
    """Tests for ACS metrics."""

    def test_acs_bailey2008(self):
        train = [["A", "B"], ["A", "B"], ["C", "D"]]
        test = [["A", "B"], ["E", "F"]]
        strengths = acs_bailey2008(train, test, n_range=(2, 2))
        assert len(strengths) == 2
        assert strengths[0] > strengths[1]

    def test_bailey_chunk_of_average_frequency_scores_half(self):
        # bigram counts: AB x3, CD x1 -> E_2 = 2.0, so S(AB) = 3/5 and S(CD) = 1/3
        train = [["A", "B"]] * 3 + [["C", "D"]]
        s_ab, s_cd, s_unseen = acs_bailey2008(train, [["A", "B"], ["C", "D"], ["X", "Y"]], n_range=(2, 2))
        assert s_ab == pytest.approx(3 / 5)
        assert s_cd == pytest.approx(1 / 3)
        assert s_unseen == 0.0

    def test_bailey_independent_of_training_repetitions(self):
        train = [["A", "B", "C"], ["A", "B", "D"], ["C", "D", "A"]]
        test = [["A", "B", "C"], ["D", "A", "B"]]
        once = acs_bailey2008(train, test)
        tenfold = acs_bailey2008(train * 10, test)
        assert once == pytest.approx(tenfold)
        # the raw-frequency Knowlton & Squire measure does scale with repetitions
        raw = global_acs_knowlton96(train, test)
        assert global_acs_knowlton96(train * 10, test) == pytest.approx([10 * g for g in raw])

    def test_bailey_bounded_and_returns_floats(self):
        train = [["A", "B", "C"]] * 50
        strengths = acs_bailey2008(train, [["A", "B", "C"], ["A"]])
        assert all(isinstance(x, float) for x in strengths)
        assert 0 <= strengths[0] < 1
        assert strengths[1] == 0.0

    def test_global_acs(self):
        train = [["A", "B", "C"]] * 10
        test = [["A", "B", "C"], ["X", "Y", "Z"]]
        strengths = global_acs_knowlton96(train, test, n_range=(2, 2))
        assert len(strengths) == 2
        assert strengths[0] > strengths[1]

    def test_anchor_acs(self):
        train = [["A", "B", "C"]] * 10
        test = [["A", "B", "C"], ["X", "Y", "Z"]]
        strengths = anchor_acs_knowlton96(train, test, n_range=(2, 2))
        assert len(strengths) == 2
        assert all(s >= 0 for s in strengths)

    def test_global_acs_builds_training_table_once_per_n(self, monkeypatch):
        from symseq.metrics.stringset import chunk_strength

        calls = []
        real = chunk_strength.chunk_ngrams_string_set
        monkeypatch.setattr(chunk_strength, "chunk_ngrams_string_set", lambda *a: calls.append(a) or real(*a))
        train = [["A", "B", "C"]] * 5
        test = [["A", "B", "C"]] * 7
        global_acs_knowlton96(train, test, n_range=(2, 3))
        assert len(calls) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
