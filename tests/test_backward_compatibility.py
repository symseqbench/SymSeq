"""Tests for backward compatibility with old API."""

import pytest
import numpy as np


class TestLegacyImports:
    """Test that old import paths still work."""

    def test_entropy_import(self):
        from symseq.metrics import entropy
        seq = ['A', 'B', 'A', 'B']
        H = entropy(seq)
        assert H > 0

    def test_compressibility_import(self):
        from symseq.metrics import compressibility
        seq = ['A'] * 50
        C = compressibility(seq)
        assert 0 < C < 1

    def test_hamming_distance_import(self):
        from symseq.metrics import hamming_distance
        d = hamming_distance(['A', 'B'], ['A', 'C'])
        assert d == 1

    def test_edit_distance_import(self):
        from symseq.metrics import edit_distance
        d = edit_distance(['A', 'B'], ['A', 'C'])
        assert d == 1

    def test_topological_entropy_import(self):
        from symseq.metrics import topological_entropy
        seq = ['A', 'B'] * 100
        lift, TE, _ = topological_entropy(sequence=seq, max_lift=3, method='lift', verbose=False)
        assert TE >= 0

    def test_count_import(self):
        from symseq.metrics import count
        seq = ['A', 'B', 'A']
        freq = count(seq, as_freq=True)
        assert freq['A'] == pytest.approx(0.667, abs=0.01)

    def test_most_common_import(self):
        from symseq.metrics import most_common
        seq = ['A'] * 5 + ['B'] * 3
        top = most_common(seq, n=1)
        assert 'A' in top

    def test_acs_imports(self):
        from symseq.metrics import (
            global_acs_knowlton96,
            anchor_acs_knowlton96,
            acs_bailey2008
        )
        train = [['A', 'B']] * 10
        test = [['A', 'B']]
        
        s1 = global_acs_knowlton96(train, test)
        s2 = anchor_acs_knowlton96(train, test)
        s3 = acs_bailey2008(train, test)
        
        assert len(s1) == 1
        assert len(s2) == 1
        assert len(s3) == 1


class TestNewImports:
    """Test that new modular imports work."""

    def test_token_module_import(self):
        from symseq.metrics import token
        assert hasattr(token, 'token_frequency')
        assert hasattr(token, 'token_duration_stats')

    def test_string_module_import(self):
        from symseq.metrics import string
        assert hasattr(string, 'entropy')
        assert hasattr(string, 'linguistic_complexity')
        assert hasattr(string, 'permutation_entropy')

    def test_stringset_module_import(self):
        from symseq.metrics import stringset
        assert hasattr(stringset, 'hamming_distance')
        assert hasattr(stringset, 'normalized_compression_distance')
        assert hasattr(stringset, 'mutual_information_strings')

    def test_grammar_module_import(self):
        from symseq.metrics import grammar
        assert hasattr(grammar, 'markov_order_selection')
        assert hasattr(grammar, 'vlmc_fit')
        assert hasattr(grammar, 'mi_decay_analysis')
        assert hasattr(grammar, 'chomsky_classification')
        assert hasattr(grammar, 'cyk_parse')


class TestFunctionalEquivalence:
    """Test that refactored functions produce same results."""

    def test_entropy_equivalence(self):
        from symseq.metrics import entropy
        from symseq.metrics.string import entropy as string_entropy
        
        seq = ['A', 'B', 'C', 'A', 'B']
        H1 = entropy(seq)
        H2 = string_entropy(seq)
        assert H1 == H2

    def test_compressibility_equivalence(self):
        from symseq.metrics import compressibility
        from symseq.metrics.string import compressibility as string_compressibility
        
        seq = ['A', 'B'] * 20
        C1 = compressibility(seq)
        C2 = string_compressibility(seq)
        assert C1 == C2

    def test_hamming_distance_equivalence(self):
        from symseq.metrics import hamming_distance
        from symseq.metrics.stringset import hamming_distance as ss_hamming
        
        seq1 = ['A', 'B', 'C']
        seq2 = ['A', 'X', 'C']
        d1 = hamming_distance(seq1, seq2)
        d2 = ss_hamming(seq1, seq2)
        assert d1 == d2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
