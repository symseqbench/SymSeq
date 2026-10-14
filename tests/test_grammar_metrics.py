"""Closed-form tests for grammar-level metrics (entropy units, MI decay, topological entropy, Markov/VLMC)."""

import itertools
import warnings

import numpy as np
import pytest
from scipy.stats import chi2

from symseq.generators.ag.inference import infer_markov, infer_vlmc
from symseq.generators.ag.synthesis import complexity
from symseq.metrics.grammar._string_sets import fixed_order_counts
from symseq.metrics.grammar.chomsky import _detect_repetition, _estimate_kolmogorov_complexity
from symseq.metrics.grammar.complexity import grammar_state_complexity
from symseq.metrics.grammar.hierarchical import (
    _mutual_information,
    grassberger_entropy,
    mi_decay_analysis,
    mi_decay_analysis_parallel,
)
from symseq.metrics.grammar.markov import context_counts, markov_order_selection
from symseq.metrics.grammar.topological import TopologicalEntropyWarning, topological_entropy
from symseq.metrics.grammar.vlmc import vlmc_fit
from symseq.utils.linalg import spectral_radius

LN2 = np.log(2)


def _iid(alphabet_size: int, n: int, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    return [str(x) for x in rng.integers(0, alphabet_size, n)]


def _markov_sequence(n: int = 600, seed: int = 1) -> list[str]:
    """Sticky 3-state chain, so MI at short distances is clearly above baseline."""
    rng = np.random.default_rng(seed)
    seq, state = [], 0
    for _ in range(n):
        seq.append("ABC"[state])
        state = state if rng.random() < 0.7 else int(rng.integers(0, 3))
    return seq


class TestGrassbergerEntropy:
    @pytest.mark.parametrize("alphabet_size, n", [(2, 1000), (4, 4000), (8, 8000)])
    def test_uniform_iid_matches_log_alphabet(self, alphabet_size, n):
        seq = _iid(alphabet_size, n)
        assert grassberger_entropy(seq, units="bits") == pytest.approx(np.log2(alphabet_size), abs=0.01)
        assert grassberger_entropy(seq, units="nats") == pytest.approx(np.log(alphabet_size), abs=0.01)

    def test_default_units_are_nats(self):
        seq = _iid(4, 2000)
        assert grassberger_entropy(seq) == grassberger_entropy(seq, units="nats")

    def test_bits_are_nats_over_ln2(self):
        seq = _iid(5, 500)
        assert grassberger_entropy(seq, units="bits") == pytest.approx(grassberger_entropy(seq, units="nats") / LN2)

    def test_invalid_units(self):
        with pytest.raises(ValueError, match="units"):
            grassberger_entropy(["a", "b"], units="dits")

    def test_empty_sequence(self):
        assert grassberger_entropy([], units="bits") == 0.0


class TestMutualInformationUnits:
    def test_mutual_information_bits_are_nats_over_ln2(self):
        seq = _markov_sequence()
        x, y = seq[:-1], seq[1:]
        assert _mutual_information(x, y, "bits") == pytest.approx(_mutual_information(x, y, "nats") / LN2)

    def test_independent_sequences_have_near_zero_mi(self):
        x, y = _iid(3, 5000, seed=1), _iid(3, 5000, seed=2)
        assert abs(_mutual_information(x, y, "bits")) < 0.01

    @pytest.mark.parametrize("analysis", [mi_decay_analysis, mi_decay_analysis_parallel])
    def test_invalid_units_fail_fast(self, analysis):
        with pytest.raises(ValueError, match="units"):
            analysis(_markov_sequence(50), max_distance=3, n_shuffles=2, units="dits")

    @pytest.mark.parametrize(
        "analysis, kwargs",
        [(mi_decay_analysis, {}), (mi_decay_analysis_parallel, {"n_jobs": 1})],
        ids=["serial", "parallel"],
    )
    def test_decay_analysis_bits_consistent_with_nats(self, analysis, kwargs):
        seq = _markov_sequence()
        common = dict(max_distance=12, n_shuffles=30, random_state=7, **kwargs)
        nats = analysis(seq, units="nats", **common)
        bits = analysis(seq, units="bits", **common)

        assert nats["units"] == "nats" and bits["units"] == "bits"
        for key in ("mi_values", "mi_adjusted", "mi_baseline", "mi_ci_lower", "mi_ci_upper"):
            np.testing.assert_allclose(bits[key], nats[key] / LN2, rtol=1e-9, atol=1e-12, err_msg=key)
        assert bits["convergence_threshold"] == pytest.approx(nats["convergence_threshold"] / LN2)
        # unit choice must not change any distance-valued conclusion
        for key in ("max_significant_distance", "max_significant_distance_ci", "convergence_distance"):
            assert bits[key] == nats[key], key
        assert bits["best_model"] == nats["best_model"]

    def test_decay_analysis_default_units_are_nats(self):
        seq = _markov_sequence(200)
        res = mi_decay_analysis(seq, max_distance=5, n_shuffles=5, random_state=0)
        assert res["units"] == "nats"


class TestMIDecayReproducibility:
    """The serial and parallel analyses use the same seeding scheme."""

    def test_serial_equals_parallel_for_same_seed(self):
        seq = _markov_sequence(300)
        kwargs = {"max_distance": 6, "n_shuffles": 20, "random_state": 7}
        serial = mi_decay_analysis(seq, **kwargs)
        parallel = mi_decay_analysis_parallel(seq, n_jobs=1, **kwargs)
        np.testing.assert_allclose(serial["mi_baseline"], parallel["mi_baseline"])
        np.testing.assert_allclose(serial["mi_adjusted"], parallel["mi_adjusted"])

    def test_serial_leaves_global_random_state_alone(self):
        seq = _markov_sequence(200)
        np.random.seed(123)
        expected = np.random.random()
        np.random.seed(123)
        mi_decay_analysis(seq, max_distance=4, n_shuffles=5, random_state=7)
        assert np.random.random() == expected


class TestTopologicalEntropy:
    @pytest.mark.parametrize("n_symbols", [2, 4, 8])
    def test_full_shift_is_log_alphabet_in_nats(self, n_symbols):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            te = topological_entropy(transitions=np.ones((n_symbols, n_symbols)), method="direct", verbose=False)
        assert te == pytest.approx(np.log(n_symbols))

    def test_acyclic_table_has_zero_entropy_and_warns(self):
        chain = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        with pytest.warns(TopologicalEntropyWarning, match="ACYCLIC"):
            assert topological_entropy(transitions=chain, method="direct", verbose=False) == 0.0

    def test_permuted_long_chain_is_detected_as_acyclic(self):
        # Not triangular, so its eigenvalues need not come out as exact zeros.
        n = 30
        perm = np.random.default_rng(0).permutation(n)
        chain = np.zeros((n, n), dtype=int)
        for a, b in itertools.pairwise(perm):
            chain[a, b] = 1
        with pytest.warns(TopologicalEntropyWarning, match="ACYCLIC"):
            assert topological_entropy(transitions=chain, method="direct", verbose=False) == 0.0

    def test_periodic_table_uses_modulus_of_eigenvalues(self):
        # 3-cycle: eigenvalues are the cube roots of unity, all of modulus 1.
        cycle = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            te = topological_entropy(transitions=cycle, method="direct", verbose=False)
        assert te == pytest.approx(0.0)

    def test_synthesis_shares_spectral_radius(self):
        ones = np.ones((4, 4))
        assert spectral_radius(ones) == pytest.approx(4.0)
        assert complexity(ones, eps_bias=1e-12) == pytest.approx(
            topological_entropy(transitions=ones, method="direct", verbose=False)
        )

    def test_lift_convergence_tolerance_is_explicit(self):
        seq = list("ABCAB" * 400)
        loose = topological_entropy(sequence=seq, max_lift=6, method="lift", verbose=False, convergence_tol=10.0)
        assert loose[0] <= 1  # any change is below the tolerance: stops at the second lift
        strict = topological_entropy(sequence=seq, max_lift=6, method="lift", verbose=False, convergence_tol=0.0)
        assert strict[0] >= loose[0]


class TestGrammarStateComplexity:
    def test_not_implemented_raises_instead_of_returning_a_number(self):
        with pytest.raises(NotImplementedError):
            grammar_state_complexity({"S": [["a"]]})


# Markov order selection and VLMC (audit G2/G3) ------------------------------------------------

_STICKY = {"A": [0.8, 0.1, 0.1], "B": [0.1, 0.8, 0.1], "C": [0.1, 0.1, 0.8]}


def _iid_string_set(n_strings=200, length=8, alphabet="ABCD", seed=0):
    rng = np.random.default_rng(seed)
    return [[str(x) for x in rng.choice(list(alphabet), size=length)] for _ in range(n_strings)]


def _order1_string_set(n_strings=200, length=8, seed=0):
    rng = np.random.default_rng(seed)
    strings = []
    for _ in range(n_strings):
        string = [str(rng.choice(list("ABC")))]
        for _ in range(length - 1):
            string.append(str(rng.choice(list("ABC"), p=_STICKY[string[-1]])))
        strings.append(string)
    return strings


class TestStringSetContract:
    @pytest.mark.parametrize("function", [markov_order_selection, vlmc_fit])
    def test_flat_stream_is_rejected(self, function):
        with pytest.raises(TypeError, match="bare str"):
            function(["A", "B", "A", "B"])

    def test_str_string_set_is_rejected(self):
        with pytest.raises(TypeError, match="not a str"):
            context_counts("ABAB", max_depth=1)

    @pytest.mark.parametrize("string_set", [[], [["A"], []]])
    def test_empty_inputs_are_rejected(self, string_set):
        with pytest.raises(ValueError):
            markov_order_selection(string_set)

    def test_eos_inside_a_string_is_rejected(self):
        with pytest.raises(ValueError, match="must not contain"):
            vlmc_fit([["A", "#"]], eos="#")

    @pytest.mark.parametrize("function", [markov_order_selection, vlmc_fit])
    def test_no_targets_is_insufficient_data(self, function):
        with pytest.raises(ValueError, match="Insufficient data"):
            function([["A"], ["B"]])

    def test_length_one_strings_have_targets_with_eos(self):
        result = markov_order_selection([["A"], ["B"]], max_order=0, eos="#")
        assert result["n_observations"] == 2

    def test_no_context_spans_two_strings(self):
        counts = context_counts([["A", "B"], ["C", "D"]], max_depth=2)["counts"]
        assert ("B", "C") not in counts
        # B ends its string: without EOS it has no target, so no B -> C transition exists.
        assert ("B",) not in counts
        assert counts[()] == {"B": 1, "D": 1}
        with_eos = context_counts([["A", "B"], ["C", "D"]], max_depth=2, eos="#")["counts"]
        assert with_eos[("B",)] == {"#": 1}
        assert with_eos[("A", "B")] == {"#": 1}

    def test_scored_targets_are_identical_across_orders(self):
        strings = [list("ABCAB"), list("CA"), list("BBBACA")]
        for eos, expected in ((None, 4 + 1 + 5), ("#", 5 + 2 + 6)):
            n = markov_order_selection(strings, max_order=4, eos=eos)["n_observations"]
            assert n == expected
            for order in range(5):
                counts, _ = fixed_order_counts([tuple(s) for s in strings], order=order, eos=eos)
                rows = [counts[()]] if order == 0 else [row for context, row in counts.items() if context]
                assert sum(sum(row.values()) for row in rows) == expected


class TestMarkovOrderSelection:
    def test_hand_computed_toy_scores(self):
        # Targets of AABABBAB are positions 2..8: A B A B B A B (3 A, 4 B), n = 7.
        result = markov_order_selection([list("AABABBAB")], max_order=2)
        ll0 = 3 * np.log(3 / 7) + 4 * np.log(4 / 7)
        ll1 = np.log(1 / 4) + 3 * np.log(3 / 4) + 2 * np.log(2 / 3) + np.log(1 / 3)
        # Order 2: the first target uses the truncated context (A,); only (A,B) is uncertain.
        ll2 = 2 * np.log(1 / 2)
        assert result["log_likelihoods"] == pytest.approx([ll0, ll1, ll2])
        assert result["n_parameters"] == [1, 2, 5]
        expected = [-2 * ll + k * np.log(7) for ll, k in zip([ll0, ll1, ll2], [1, 2, 5], strict=True)]
        assert result["scores"] == pytest.approx(expected)
        assert result["optimal_order"] == 0
        aic = markov_order_selection([list("AABABBAB")], max_order=2, criterion="AIC")
        assert aic["scores"] == pytest.approx(
            [-2 * ll + 2 * k for ll, k in zip([ll0, ll1, ll2], [1, 2, 5], strict=True)]
        )

    def test_recovers_order_zero_and_one(self):
        assert markov_order_selection(_iid_string_set(alphabet="ABC"), max_order=3)["optimal_order"] == 0
        result = markov_order_selection(_order1_string_set(), max_order=3)
        assert result["optimal_order"] == 1
        assert not result["selected_at_cap"]
        for row in result["transition_probabilities"].values():
            assert sum(row.values()) == pytest.approx(1.0)

    def test_selected_at_cap_is_reported(self):
        result = markov_order_selection([["A", "B"] * 10 for _ in range(20)], max_order=1)
        assert result["optimal_order"] == 1
        assert result["selected_at_cap"]

    def test_max_order_beyond_strings_warns_and_is_not_lowered(self):
        with pytest.warns(UserWarning, match="longest usable context"):
            result = markov_order_selection([["A", "B"], ["B", "A"]] * 5, max_order=3)
        assert len(result["scores"]) == 4

    def test_constant_input_selects_order_zero(self):
        result = markov_order_selection([["A"] * 5] * 3, max_order=2)
        assert result["optimal_order"] == 0
        assert result["scores"] == [0.0, 0.0, 0.0]

    @pytest.mark.parametrize("kwargs", [{"max_order": -1}, {"max_order": 1.5}, {"criterion": "HQ"}])
    def test_invalid_settings(self, kwargs):
        with pytest.raises(ValueError):
            markov_order_selection([["A", "B"]], **kwargs)

    def test_inference_uses_the_metric_scorer(self):
        strings = [["A", "B", "X"]] * 100 + [["C", "B", "Y"]] * 100
        inference = infer_markov(strings, order="bic", max_order=2).metadata["inference"]
        metric = markov_order_selection(strings, max_order=2, eos="#")
        assert inference["bic_scores"] == dict(enumerate(metric["scores"]))
        # Pinned values: inference output must not change across refactors.
        assert inference["bic_scores"] == pytest.approx(
            {0: 1627.5782669017906, 1: 437.1821136043818, 2: 191.9078896564844}
        )
        assert inference["order"] == metric["optimal_order"] == 2


class TestVlmcFit:
    def test_uninformative_extensions_are_pruned(self):
        fit = vlmc_fit([["A", "B", "X"]] * 100 + [["C", "B", "X"]] * 100, max_depth=2)
        assert fit["contexts"] == ((), ("A",), ("B",), ("C",))

    def test_informative_extensions_are_retained(self):
        fit = vlmc_fit([["A", "B", "X"]] * 100 + [["C", "B", "Y"]] * 100, max_depth=2)
        assert {("A", "B"), ("C", "B")} <= set(fit["terminal_contexts"])
        assert fit["context_tree"][("A", "B")] == {"X": 1.0}

    def test_independent_source_gives_root_only(self):
        fit = vlmc_fit(_iid_string_set(), max_depth=4)
        assert fit["contexts"] == ((),)
        assert fit["terminal_contexts"] == fit["predictive_contexts"] == ((),)
        assert fit["max_depth_used"] == 0
        assert sum(fit["context_tree"][()].values()) == pytest.approx(1.0)

    def test_order_one_source_gives_depth_one_terminals(self):
        fit = vlmc_fit(_order1_string_set(), max_depth=4)
        assert fit["terminal_contexts"] == (("A",), ("B",), ("C",))

    def test_internal_nodes_can_be_predictive(self):
        # (C, B) predicts like (B,) and is pruned, so histories ending in C B fall back to (B,).
        fit = vlmc_fit([["A", "B", "Y"]] * 40 + [["C", "B", "X"]] * 100 + [["C", "B", "Y"]] * 100, max_depth=2)
        assert ("B",) in fit["predictive_contexts"]
        assert ("B",) not in fit["terminal_contexts"]
        assert fit["num_nodes"] == len(fit["contexts"])

    def test_cutoff_rules(self):
        strings = _order1_string_set(n_strings=50)
        n = sum(len(s) - 1 for s in strings)
        assert vlmc_fit(strings)["cutoff"] == pytest.approx(2 * np.log(n) / 2)
        chi = vlmc_fit(strings, alpha=0.05)
        assert chi["cutoff_rule"] == "chi2"
        assert chi["cutoff"] == pytest.approx(chi2.ppf(0.95, df=2) / 2)
        with_eos = vlmc_fit(strings, eos="#")
        assert with_eos["cutoff"] == pytest.approx(3 * np.log(n + len(strings)) / 2)
        assert vlmc_fit(strings, cutoff=1.5)["cutoff_rule"] == "explicit"

    def test_constant_input_gives_root_only(self):
        fit = vlmc_fit([["A"] * 5] * 3)
        assert fit["contexts"] == ((),)
        assert fit["cutoff"] == np.inf

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"alpha": 0.05, "cutoff": 2.0},
            {"min_count": 1},
            {"max_depth": 0},
            {"alpha": 1.5},
            {"cutoff": -1.0},
            {"cutoff": "aic"},
        ],
    )
    def test_invalid_settings(self, kwargs):
        with pytest.raises(ValueError):
            vlmc_fit([["A", "B"]] * 3, **kwargs)

    def test_legacy_pruning_threshold_is_rejected(self):
        with pytest.raises(TypeError):
            vlmc_fit([["A", "B"]] * 3, pruning_threshold=0.01)

    def test_inference_uses_the_metric_fit(self):
        strings = [["A", "B", "X"]] * 100 + [["C", "B", "Y"]] * 100
        inference = infer_vlmc(strings, max_depth=2).metadata["inference"]
        metric = vlmc_fit(strings, max_depth=2, alpha=0.05, eos="#")
        assert inference["contexts"] == metric["contexts"]
        assert inference["discrepancies"] == metric["discrepancies"]
        assert inference["cutoff"] == metric["cutoff"] == pytest.approx(chi2.ppf(0.95, df=5) / 2)
        assert inference["discrepancies"][("A", "B")] == pytest.approx(100 * np.log(2))


class TestChomskyMultiCharSymbols:
    """Symbols of more than one character must not collide with their concatenation (audit T5)."""

    def test_repetition_units_are_not_flattened(self):
        # flattened, every pair of symbols reads 'ABC' and the sequence looks perfectly periodic
        seq = ["A", "BC", "AB", "C", "AB", "C", "A", "BC"]
        assert _detect_repetition(seq) < 1.0
        assert _detect_repetition(["A", "BC"] * 10) == 1.0

    def test_kolmogorov_complexity_separates_collisions(self):
        regular = ["A", "BC"] * 50
        irregular = ["A", "BC", "AB", "C", "ABC", "A"] * 17
        assert _estimate_kolmogorov_complexity(regular) < _estimate_kolmogorov_complexity(irregular)

    def test_kolmogorov_rejects_string_set(self):
        with pytest.raises(TypeError, match="single sequence"):
            _estimate_kolmogorov_complexity([["A", "B"]] * 5)
