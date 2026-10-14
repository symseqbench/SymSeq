"""Tests for string-level metrics."""

import logging
import os
import subprocess
import sys

import numpy as np
import pytest

from symseq.metrics.string import (
    block_entropy,
    compressibility,
    compute_compression_metrics_ensemble,
    compute_entropy_metrics_ensemble,
    detect_saturation_point,
    emc,
    entropy,
    entropy_rate,
    estimate_emc,
    linguistic_complexity,
    lzw_complexity,
    permutation_entropy,
)
from symseq.metrics.string.compression import encode_symbols
from symseq.utils.strtools import chunk_ngrams_string, chunk_transitions


class TestEntropy:
    """Tests for entropy functions."""

    def test_uniform_distribution(self):
        seq = ["A", "B", "C", "D"]
        H = entropy(seq, bias_correction=False)
        assert pytest.approx(2.0, abs=0.01) == H

    def test_constant_sequence(self):
        seq = ["A"] * 10
        H = entropy(seq, bias_correction=False)
        assert pytest.approx(0.0, abs=0.01) == H

    def test_binary_sequence(self):
        seq = ["A", "B"]
        H = entropy(seq, bias_correction=False)
        assert pytest.approx(1.0, abs=0.01) == H

    def test_empty_sequence(self):
        seq = []
        H = entropy(seq)
        assert H == 0.0

    def test_bias_correction_increases_entropy(self):
        """Test that Miller-Madow bias correction increases entropy estimate."""
        seq = ["A", "B", "C", "D"]
        H_uncorrected = entropy(seq, bias_correction=False)
        H_corrected = entropy(seq, bias_correction=True)
        # Corrected entropy should be higher due to bias correction
        assert H_corrected > H_uncorrected
        # Verify the correction magnitude is reasonable (< 1 bit for this sample)
        assert (H_corrected - H_uncorrected) < 1.0

    def test_bias_correction_formula(self):
        """Test that bias correction matches Miller-Madow formula."""
        import numpy as np

        seq = ["A", "B", "C", "D"]
        H_uncorrected = entropy(seq, bias_correction=False)
        H_corrected = entropy(seq, bias_correction=True)

        # Manual calculation: correction = (M - 1) / (2 * N * ln(2))
        M = 4  # number of unique symbols
        N = 4  # sequence length
        expected_correction = (M - 1) / (2.0 * N * np.log(2))

        assert H_corrected == pytest.approx(H_uncorrected + expected_correction, abs=0.001)


class TestBlockEntropy:
    """Tests for block_entropy function."""

    def test_block_size_1(self):
        seq = ["A", "B", "A", "B"]
        H = block_entropy(seq, block_size=1, bias_correction=False)
        H_regular = entropy(seq, bias_correction=False)
        assert pytest.approx(H_regular, abs=0.01) == H

    def test_block_size_2(self):
        seq = ["A", "B"] * 10
        H = block_entropy(seq, block_size=2)
        assert H > 0

    def test_sequence_too_short(self):
        seq = ["A", "B"]
        H = block_entropy(seq, block_size=5)
        assert H == 0.0


class TestEntropyRate:
    """Tests for entropy_rate function."""

    def test_iid_sequence(self):
        np.random.seed(42)
        seq = list(np.random.choice(["A", "B"], size=100))
        h_mu = entropy_rate(seq, max_block_size=3)
        H = entropy(seq)
        assert h_mu <= H

    def test_deterministic_sequence(self):
        seq = ["A", "B"] * 50
        h_mu = entropy_rate(seq, max_block_size=3)
        assert h_mu >= 0


class TestEMC:
    """Tests for Effective Measure Complexity."""

    def test_iid_sequence(self):
        np.random.seed(42)
        seq = list(np.random.choice(["A", "B"], size=100))
        emc_val, h_mu = emc(seq, max_block_size=5)
        assert emc_val >= 0
        assert h_mu >= 0

    def test_deterministic_sequence(self):
        seq = ["A"] * 50
        emc_val, _h_mu = emc(seq, max_block_size=5)
        assert emc_val >= 0


class TestCompressibility:
    """Tests for compressibility function."""

    def test_repetitive_sequence(self):
        seq = ["A"] * 100
        C = compressibility(seq)
        assert 0 < C < 0.5

    def test_random_sequence(self):
        np.random.seed(42)
        seq = list(np.random.choice(["A", "B", "C", "D"], size=100))
        C = compressibility(seq)
        assert 0 < C <= 1.0

    def test_range(self):
        seq = ["A", "B", "C"] * 10
        C = compressibility(seq)
        assert 0 < C <= 1.0

    def test_short_strings_not_dominated_by_gzip_framing(self):
        assert compressibility(["A"] * 8) < 1.0
        assert compressibility(["A"] * 10) < 1.0

    def test_framing_is_content_independent(self):
        from symseq.metrics.string.compression import gzip_payload_size

        assert gzip_payload_size(b"") == 0
        assert 0 < gzip_payload_size(b"ABCDEFGH") <= 8


class TestLZWComplexity:
    """Tests for LZW complexity."""

    def test_normalized(self):
        seq = ["A", "B", "A", "B"] * 10
        C = lzw_complexity(seq, normalized=True)
        assert 0 < C <= 1.0

    def test_unnormalized(self):
        seq = ["A", "B", "A", "B"] * 10
        C = lzw_complexity(seq, normalized=False)
        assert C > 0
        assert isinstance(C, (int, float))


class TestLZWMultiCharacterSymbols:
    def test_multichar_symbols_do_not_crash(self):
        assert lzw_complexity(["AB", "CD"] * 10) == pytest.approx(lzw_complexity(["a", "b"] * 10))

    def test_non_latin1_symbols(self):
        assert lzw_complexity(["α", "β", "α", "β"]) == lzw_complexity(["a", "b", "a", "b"])

    def test_indexed_states(self):
        assert 0 < lzw_complexity(["A(1)", "B(2)", "A(1)", "C(3)"] * 5) <= 1


class TestMultiCharacterSymbols:
    """Symbols of more than one character must not collide with their concatenation (audit T5)."""

    def test_ngrams_are_tuples_and_distinct(self):
        all_a, uniq_a = chunk_ngrams_string(["A", "BC", "A", "BC"], 2)
        all_b, uniq_b = chunk_ngrams_string(["AB", "C", "AB", "C"], 2)
        assert all_a == [("A", "BC"), ("BC", "A"), ("A", "BC")]
        assert set(uniq_a).isdisjoint(uniq_b)
        assert all(len(g) == 2 for g in all_a + all_b)

    def test_unique_ngrams_in_first_occurrence_order(self):
        _, uniq = chunk_ngrams_string(list("BABAB"), 2)
        assert uniq == [("B", "A"), ("A", "B")]

    def test_chunk_transitions_multichar(self):
        seq = ["A(1)", "B(2)", "A(1)", "C(3)"] * 3
        M = chunk_transitions(seq, 2, verbose=False)
        assert M.shape == (4, 4)
        assert M.sum() == 4  # 4 distinct bigram transitions (cyclic sequence)

    def test_chunk_transitions_dataframe_labels(self):
        df = chunk_transitions(["A", "BC", "A", "BC"], 2, verbose=False, as_dataframe=True)
        assert sorted(df.columns) == ["A BC", "BC A"]

    def test_chunk_transitions_too_short(self):
        with pytest.raises(ValueError, match="shorter than chunk_len"):
            chunk_transitions(["A"], 2, verbose=False)

    def test_encode_symbols_is_injective(self):
        a, b = encode_symbols(["A", "BC"], ["AB", "C"])
        assert a != b

    def test_encode_symbols_shared_alphabet(self):
        a, b = encode_symbols(["A", "B"], ["B", "A"])
        assert a == bytes([0, 1]) and b == bytes([1, 0])

    def test_encode_symbols_width_grows_with_alphabet(self):
        (small,) = encode_symbols([str(i) for i in range(256)])
        (large,) = encode_symbols([str(i) for i in range(257)])
        assert len(small) == 256 and len(large) == 2 * 257

    def test_compressibility_rejects_string_set(self):
        with pytest.raises(TypeError, match="single sequence"):
            compressibility([["A", "B"], ["A", "B"]])


class TestCompressionEnsemble:
    def _ensemble(self, n_jobs):
        # integer symbols make compressibility fail (str.join) while lzw still works
        return compute_compression_metrics_ensemble([["A", "B"] * 10, [1, 2] * 10], n_jobs=n_jobs)

    @pytest.mark.parametrize("n_jobs", [1, 2])
    def test_failure_is_nan_and_aligned_and_warned(self, n_jobs):
        with pytest.warns(UserWarning, match="failed for 1 of 2"):
            res = self._ensemble(n_jobs)
        assert len(res["compressibility"]) == len(res["lzw_complexity"]) == 2
        assert not np.isnan(res["compressibility"][0]) and np.isnan(res["compressibility"][1])
        assert not np.isnan(res["lzw_complexity"][1])

    def test_serial_equals_parallel(self):
        with pytest.warns(UserWarning):
            serial = self._ensemble(1)
        with pytest.warns(UserWarning):
            parallel = self._ensemble(2)
        np.testing.assert_array_equal(serial["compressibility"], parallel["compressibility"])
        np.testing.assert_array_equal(serial["lzw_complexity"], parallel["lzw_complexity"])


class TestSaturationDetection:
    def test_constant_sequence_has_no_memory(self):
        res = detect_saturation_point(["A"] * 200, max_k=6)
        assert res["k_star"] == 1

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown saturation method"):
            detect_saturation_point(list("ABCABC") * 20, max_k=4, method="not-a-method")

    def test_ensemble_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown saturation method"):
            compute_entropy_metrics_ensemble([list("ABCABC") * 20], saturation_method="not-a-method")

    @pytest.mark.parametrize("method", ["adaptive", "relative", "statistical", "hybrid"])
    def test_ensemble_applies_the_requested_method(self, method):
        rng = np.random.default_rng(0)
        seq = [str(x) for x in rng.integers(0, 3, 400)]
        expected = detect_saturation_point(seq, 6, method=method)["k_star"]
        res = compute_entropy_metrics_ensemble([seq], max_block_size=6, saturation_method=method)
        assert res["saturation_results_raw"][0]["k_star"] == expected
        assert res["saturation_results_raw"][0]["method_used"] == method


def _iid(n, alphabet_size, seed):
    rng = np.random.default_rng(seed)
    return [str(x) for x in rng.integers(0, alphabet_size, n)]


def _golden_mean(n, seed):
    """Order-1 binary chain: '0' is always followed by '1'. h = 2/3, E = H(1/3) - 2/3 bits."""
    rng = np.random.default_rng(seed)
    seq = ["1"]
    for _ in range(n - 1):
        seq.append("1" if seq[-1] == "0" or rng.random() < 0.5 else "0")
    return seq


def _noisy_xor(n, flip, seed):
    """Order-2 binary chain: x_t = x_{t-1} XOR x_{t-2}, flipped with probability `flip`."""
    rng = np.random.default_rng(seed)
    x = [0, 1]
    for _ in range(n - 2):
        x.append((x[-1] ^ x[-2]) ^ int(rng.random() < flip))
    return [str(v) for v in x]


def _binary_entropy(p):
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


class TestEMCEstimator:
    """Closed-form checks for estimate_emc and the functions built on it (audit finding T1)."""

    def test_iid_has_no_excess_entropy(self):
        res = estimate_emc(_iid(2000, 3, seed=0))
        assert res["order"] == 0
        assert res["emc"] < 0.05
        assert res["entropy_rate"] == pytest.approx(np.log2(3), abs=0.03)
        assert res["bias_ok"]

    def test_emc_defaults_on_short_iid(self):
        # Regression: the old estimator returned E = 6.6 bits here (truth 0)
        E, h = emc(_iid(400, 3, seed=1))
        assert E < 0.1
        assert h == pytest.approx(np.log2(3), abs=0.05)

    def test_golden_mean(self):
        res = estimate_emc(_golden_mean(5000, seed=0))
        # Finite-sample selection can overfit. Check the actual estimates, not nesting alone.
        assert res["order"] >= 1
        assert res["entropy_rate"] == pytest.approx(2 / 3, abs=0.02)
        assert res["emc"] == pytest.approx(_binary_entropy(1 / 3) - 2 / 3, abs=0.03)

    @pytest.mark.parametrize("period", [2, 3, 5])
    def test_periodic_sequence(self, period):
        res = estimate_emc([str(i % period) for i in range(300)])
        assert res["order"] == 1
        assert res["entropy_rate"] == 0.0
        assert res["emc"] == pytest.approx(np.log2(period))

    def test_order_two_chain_without_decomposition_warning(self, caplog):
        flip = 0.1
        seq = _noisy_xor(5000, flip, seed=0)
        with caplog.at_level(logging.WARNING):
            E, h = emc(seq)
        assert caplog.text == ""
        assert estimate_emc(seq)["order"] == 2
        assert h == pytest.approx(_binary_entropy(flip), abs=0.02)
        assert pytest.approx(2 - 2 * _binary_entropy(flip), abs=0.05) == E  # E = H_2 - 2h, H_2 = 2 bits

    def test_constant_sequence(self):
        res = estimate_emc(["A"] * 100)
        assert (res["order"], res["emc"], res["entropy_rate"]) == (0, 0.0, 0.0)

    def test_fixed_order_is_used_exactly(self):
        res = estimate_emc(_golden_mean(1000, seed=0), order=3)
        assert res["order"] == 3
        assert res["order_method"] == "fixed"
        assert res["bic_scores"] is None

    def test_bias_estimate_flags_undersampled_order(self):
        seq = _iid(2000, 2, seed=0)
        res = estimate_emc(seq, order=7)
        # True E is 0, so the whole estimate is bias; the diagnostic should predict it
        assert res["emc"] == pytest.approx(res["emc_bias_estimate"], abs=0.05)
        assert not res["bias_ok"]
        assert estimate_emc(seq)["bias_ok"]

    @pytest.mark.parametrize("order", [-1, 1.5, "aic", True])
    def test_invalid_order_raises(self, order):
        with pytest.raises(ValueError, match="order must be"):
            estimate_emc(list("AB") * 20, order=order)

    def test_order_too_high_for_sequence_raises(self):
        with pytest.raises(ValueError, match="supports at most order 4"):
            estimate_emc(list("ABABAB"), order=5)

    def test_too_short_sequence_raises(self):
        with pytest.raises(ValueError, match="At least two symbols"):
            estimate_emc(["A"])

    def test_multi_character_symbols_do_not_merge(self):
        # ('AB', 'C') and ('A', 'BC') would both become 'ABC' if blocks were joined into strings
        seq = ["AB", "C", "A", "BC"] * 50
        assert block_entropy(seq, 2) == pytest.approx(np.log2(4), abs=0.01)

    def test_all_entry_points_agree(self):
        seq = _golden_mean(800, seed=3)
        ref = estimate_emc(seq, max_order=9)
        assert emc(seq, max_block_size=10) == (ref["emc"], ref["entropy_rate"])
        assert entropy_rate(seq, max_block_size=10) == ref["entropy_rate"]
        ens = compute_entropy_metrics_ensemble([seq], max_block_size=10)
        assert ens["emc"] == [ref["emc"]]
        assert ens["entropy_rate"] == [ref["entropy_rate"]]
        assert ens["emc_order"] == [ref["order"]]

    def test_ensemble_fixed_order_matches_estimate_emc(self):
        seqs = [_golden_mean(500, seed=s) for s in range(3)]
        ens = compute_entropy_metrics_ensemble(seqs, emc_order=2)
        assert ens["emc_order_method"] == "fixed"
        assert ens["emc"] == [estimate_emc(s, order=2)["emc"] for s in seqs]

    def test_ensemble_lists_stay_aligned(self):
        seqs = [_golden_mean(500, seed=0), list("ABABA")]
        ens = compute_entropy_metrics_ensemble(seqs, min_sequence_length=2, emc_order=5)
        for key in ("block_entropy_k2", "entropy_rate", "emc", "emc_order", "emc_bias_estimate", "emc_bias_ok"):
            assert len(ens[key]) == 2, key
        assert np.isnan(ens["emc"][1])
        assert ens["emc_bias_ok"][1] is None

    def test_ensemble_invalid_order_raises(self):
        with pytest.raises(ValueError, match="order must be"):
            compute_entropy_metrics_ensemble([list("AB") * 20], emc_order="aic")

    def test_excess_curve(self):
        seq = _golden_mean(1000, seed=0)
        _, h, curve = emc(seq, max_block_size=6, return_curve=True)
        assert len(curve) == 6
        assert curve[0] == pytest.approx(block_entropy(seq, 1) - h)


class TestEMCDiagnostics:
    """Behavioral checks for finite-resolution results and the sampling policy."""

    def test_short_auto_preserves_empirical_entropy(self):
        seq = [str(i) for i in range(10)] * 4
        res = estimate_emc(seq)
        assert res["status"] == "insufficient_data"
        assert res["entropy"] == pytest.approx(np.log2(10))
        for key in ("emc", "entropy_rate", "order", "bias_ok", "n_scored", "max_order_searched"):
            assert res[key] is None, key
        assert "sampling_screen" in res["search_limit_reasons"]
        assert np.isnan(emc(seq)[0])
        assert np.isnan(entropy_rate(seq, 10))

    def test_no_order_passes_screen(self):
        res = estimate_emc(list("AB"))
        assert res["sampling_order_limit"] == -1
        assert res["status"] == "insufficient_data"
        assert res["entropy"] == 1.0

    def test_zero_order_is_explicit(self):
        seq = list("0011") * 100
        for res in (estimate_emc(seq, order=0), estimate_emc(seq, max_order=0)):
            assert res["order"] == 0
            assert res["emc"] == 0
            assert res["entropy_rate"] == 1
            assert res["status"] == "computed"
            assert "truncation_not_assessed" in res["limitations"]
        assert estimate_emc(seq, max_order=0)["max_order_searched"] == 0

    def test_fixed_order_retained_when_sparse(self):
        seq = _iid(100, 3, 0)
        res = estimate_emc(seq, order=5, max_order=0)
        assert res["status"] == "sparse"
        assert res["order"] == 5
        assert res["emc"] is not None
        assert len(res["block_entropies"]) == 6
        for key in ("bic_scores", "max_order_searched", "search_order_limit", "n_scored", "order_at_search_limit"):
            assert res[key] is None
        assert "sampling_screen_failed" in res["warnings"]

    def test_short_search_keeps_observations(self):
        res = estimate_emc(_iid(12, 2, 0))
        assert res["max_order_searched"] <= 1
        assert res["n_scored"] >= 11

    def test_common_positions_and_runtime_cap(self):
        seq = list("AB") * 200
        res = estimate_emc(seq, max_order=100)
        assert res["max_order_searched"] == 30
        assert res["n_scored"] == len(seq) - 30
        assert "runtime_cap" in res["search_limit_reasons"]
        assert set(res["bic_scores"]) == set(range(31))

    def test_default_entry_points_agree(self):
        seq = _golden_mean(800, 3)
        ref = estimate_emc(seq)
        assert emc(seq) == (ref["emc"], ref["entropy_rate"])
        ens = compute_entropy_metrics_ensemble([seq])
        record = ens["records"][0]
        for key in ("emc", "entropy_rate", "order", "status", "bias_ok", "n_scored", "max_order_searched"):
            assert record[key] == ref[key], key

    def test_ensemble_keeps_indices_and_unavailable_results(self):
        seqs = [["A"], ["A", "B"], list("AABAB") * 60]
        result = compute_entropy_metrics_ensemble(seqs, min_sequence_length=2)
        assert result["input_indices"] == [1, 2]
        assert result["block_entropy_k2"][0] == 0
        assert result["entropy"][0] == 1
        assert result["emc_status"] == ["insufficient_data", "computed"]
        assert result["records"][0]["emc"] is None
        assert np.isnan(result["emc"][0])
        assert result["records"][0]["saturation"] is None
        assert result["records"][1]["saturation"] == result["saturation_results_raw"][0]

    def test_curve_extends_beyond_screen(self):
        seq = _iid(100, 3, 0)
        E, h, curve = emc(seq, max_block_size=10, return_curve=True)
        assert len(curve) == 10
        assert curve == pytest.approx([block_entropy(seq, L) - L * h for L in range(1, 11)])
        assert estimate_emc(seq, max_order=9)["emc"] == E

    def test_saturation_is_independent_of_emc(self):
        seq = _iid(400, 3, 0)
        first = compute_entropy_metrics_ensemble([seq], max_block_size=6, saturation_method="adaptive")
        second = compute_entropy_metrics_ensemble([seq], max_block_size=6, saturation_method="relative")
        assert first["emc"] == second["emc"]
        assert first["entropy_rate"] == second["entropy_rate"]
        assert len(first["records"][0]["block_entropies"]) >= 6
        assert first["saturation_results_raw"][0]["k_star"] == detect_saturation_point(seq, 6)["k_star"]
        assert first["saturation"]["confidence_calibrated"] is False
        assert first["records"][0]["saturation"]["interpretation"] == "empirical_flattening"

    def test_repeated_symbol_period(self):
        res = estimate_emc(list("AABAB") * 60)
        assert res["order"] == 4
        assert res["entropy_rate"] == 0
        assert res["emc"] == pytest.approx(np.log2(5), abs=1e-3)
        assert res["entropy_rate_bias_estimate"] == 0

    def test_lagged_memory_and_capped_search(self):
        rng = np.random.default_rng(0)
        x = rng.integers(0, 2, 20000)
        for t in range(10, len(x)):
            x[t] = x[t - 10] ^ int(rng.random() < 0.1)
        seq = x.astype(str).tolist()
        found = estimate_emc(seq)
        capped = estimate_emc(seq, max_order=9)
        assert found["order"] == 10
        assert found["entropy_rate"] == pytest.approx(_binary_entropy(0.1), abs=0.06)
        assert capped["order"] < 9
        assert capped["order_at_search_limit"] is False
        assert capped["emc"] < 0.05
        assert "truncation_not_assessed" in capped["limitations"]

    def test_bias_ok_does_not_claim_confidence(self):
        # The fixed order intentionally misses the phase information in this source.
        res = estimate_emc(list("0011") * 100, order=0)
        assert res["bias_ok"] is True
        assert res["emc"] == 0
        assert "sampling_uncertainty_not_quantified" in res["limitations"]
        assert "asymptotic_convergence_not_established" in res["limitations"]
        assert "reliable" not in res

    @pytest.mark.parametrize("value", [-1, np.nan, np.inf, True, "0.1"])
    def test_invalid_bias_tolerance(self, value):
        with pytest.raises(ValueError, match="bias_tolerance"):
            estimate_emc(list("AB") * 20, bias_tolerance=value)
        with pytest.raises(ValueError, match="bias_tolerance"):
            compute_entropy_metrics_ensemble([], bias_tolerance=value)

    @pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, True, "5"])
    def test_invalid_sampling_threshold(self, value):
        with pytest.raises(ValueError, match="min_obs_per_context"):
            estimate_emc(list("AB") * 20, min_obs_per_context=value)
        with pytest.raises(ValueError, match="min_obs_per_context"):
            compute_entropy_metrics_ensemble([], min_obs_per_context=value)

    def test_invalid_caps_and_method(self):
        with pytest.raises(ValueError, match="max_order"):
            estimate_emc(list("AB") * 20, max_order=True)
        with pytest.raises(ValueError, match="max_block_size"):
            compute_entropy_metrics_ensemble([], max_block_size=0)
        with pytest.raises(ValueError, match="method"):
            entropy_rate(list("AB") * 20, 6, method="bogus")

    def test_normalized_constant(self):
        assert entropy_rate(["A"] * 20, 6, method="normalized") == 0


class TestLinguisticComplexity:
    """Tests for linguistic_complexity function."""

    def test_uniform_sequence(self):
        seq = ["A", "B", "C", "D"]
        C = linguistic_complexity(seq)
        assert pytest.approx(1.0, abs=0.01) == C

    def test_repetitive_sequence(self):
        seq = ["A"] * 10
        C = linguistic_complexity(seq)
        assert pytest.approx(1.0, abs=0.01) == C

    def test_empty_sequence(self):
        seq = []
        C = linguistic_complexity(seq)
        assert C == 0.0

    def test_range(self):
        seq = ["A", "B", "C"] * 5
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
        seq = ["A", "B", "C"] * 30
        H = permutation_entropy(seq, order=3, normalize=True)
        assert 0 <= H <= 1.0

    def test_range(self):
        seq = np.random.randn(100)
        H = permutation_entropy(seq, order=3, normalize=True)
        assert 0 <= H <= 1.0

    def test_deterministic_across_hash_seeds(self):
        code = (
            "from symseq.metrics.string import permutation_entropy as p;"
            "s = list('ABCAACBBCABCCABACBACBCCA' * 10);"
            "print(repr(p(s, order=3)))"
        )
        outputs = {
            subprocess.run(
                [sys.executable, "-c", code],
                env={**os.environ, "PYTHONHASHSEED": seed},
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            for seed in ("1", "2", "3")
        }
        assert len(outputs) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
