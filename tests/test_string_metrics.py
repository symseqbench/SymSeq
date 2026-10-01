"""Tests for string-level metrics."""

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
    linguistic_complexity,
    lzw_complexity,
    permutation_entropy,
)


class TestEntropy:
    """Tests for entropy functions."""

    def test_uniform_distribution(self):
        seq = ["A", "B", "C", "D"]
        H = entropy(seq, bias_correction=False)
        assert H == pytest.approx(2.0, abs=0.01)

    def test_constant_sequence(self):
        seq = ["A"] * 10
        H = entropy(seq, bias_correction=False)
        assert H == pytest.approx(0.0, abs=0.01)

    def test_binary_sequence(self):
        seq = ["A", "B"]
        H = entropy(seq, bias_correction=False)
        assert H == pytest.approx(1.0, abs=0.01)

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
        assert H == pytest.approx(H_regular, abs=0.01)

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
        emc_val, h_mu = emc(seq, max_block_size=5)
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


class TestLinguisticComplexity:
    """Tests for linguistic_complexity function."""

    def test_uniform_sequence(self):
        seq = ["A", "B", "C", "D"]
        C = linguistic_complexity(seq)
        assert C == pytest.approx(1.0, abs=0.01)

    def test_repetitive_sequence(self):
        seq = ["A"] * 10
        C = linguistic_complexity(seq)
        assert C == pytest.approx(1.0, abs=0.01)

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
