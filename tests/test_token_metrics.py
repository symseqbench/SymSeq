"""Tests for token-level metrics."""

import pytest
import numpy as np

from symseq.metrics.token import token_frequency, most_common_tokens, token_duration_stats


class TestTokenFrequency:
    """Tests for token_frequency function."""

    def test_basic_frequency(self):
        seq = ['A', 'B', 'A', 'C']
        freq = token_frequency(seq, as_freq=True)
        assert freq['A'] == 0.5
        assert freq['B'] == 0.25
        assert freq['C'] == 0.25

    def test_counts(self):
        seq = ['A', 'B', 'A', 'C']
        counts = token_frequency(seq, as_freq=False)
        assert counts['A'] == 2
        assert counts['B'] == 1
        assert counts['C'] == 1

    def test_empty_sequence(self):
        seq = []
        freq = token_frequency(seq)
        assert freq == {}

    def test_single_token(self):
        seq = ['A']
        freq = token_frequency(seq)
        assert freq['A'] == 1.0


class TestMostCommonTokens:
    """Tests for most_common_tokens function."""

    def test_top_n(self):
        seq = ['A'] * 5 + ['B'] * 3 + ['C'] * 1
        top = most_common_tokens(seq, n=2, as_freq=True)
        assert len(top) == 2
        assert 'A' in top
        assert 'B' in top
        assert top['A'] > top['B']

    def test_n_greater_than_unique(self):
        seq = ['A', 'B']
        top = most_common_tokens(seq, n=10)
        assert len(top) == 2


class TestTokenDurationStats:
    """Tests for token_duration_stats function."""

    def test_explicit_durations(self):
        seq = ['A', 'B', 'A', 'B']
        durs = [0.5, 1.0, 0.7, 0.9]
        stats = token_duration_stats(seq, durations=durs, summary_stats=False)
        assert 'A' in stats
        assert 'B' in stats
        assert len(stats['A']) == 2
        assert len(stats['B']) == 2

    def test_explicit_durations_summary(self):
        seq = ['A', 'B', 'A', 'B']
        durs = [0.5, 1.0, 0.7, 0.9]
        stats = token_duration_stats(seq, durations=durs, summary_stats=True)
        assert stats['A'][0] == pytest.approx(0.6, abs=0.01)
        assert stats['B'][0] == pytest.approx(0.95, abs=0.01)

    def test_frame_based(self):
        seq_frames = ['A', 'A', 'A', 'B', 'B', 'A']
        stats = token_duration_stats(seq_frames, frame_rate=30.0, summary_stats=False)
        assert len(stats['A']) == 2
        assert len(stats['B']) == 1
        assert stats['A'][0] == pytest.approx(0.1, abs=0.01)

    def test_missing_parameters(self):
        seq = ['A', 'B']
        with pytest.raises(ValueError):
            token_duration_stats(seq)

    def test_duration_length_mismatch(self):
        seq = ['A', 'B', 'C']
        durs = [0.5, 1.0]
        with pytest.raises(ValueError):
            token_duration_stats(seq, durations=durs)

    def test_empty_sequence(self):
        seq = []
        stats = token_duration_stats(seq, frame_rate=30.0)
        assert stats == {}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
