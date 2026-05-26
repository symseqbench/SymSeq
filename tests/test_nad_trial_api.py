"""
test_nad_trial_api.py

Tests for the Trial-based API of NonAdjacentDependencies.
"""

import numpy as np
import pytest

from symseq.generators.nad import NonAdjacentDependencies


def _nad(seed=42):
    return NonAdjacentDependencies(n_deps=3, n_unique_fillers=2, seed=seed, verbose=False)


class TestTrialAPI:
    def test_generate_trial_returns_trial(self):
        from symseq.trial import Trial, Target
        gen = _nad()
        trial = gen.generate_trial(filler_len=2)
        assert isinstance(trial, Trial)
        assert len(trial.symbols) == 4  # A_i + 2 fillers + B_i
        assert trial.states is None
        assert isinstance(trial.targets["grammaticality"], Target)
        assert isinstance(trial.targets["pair_index"], Target)

    def test_grammaticality_always_true(self):
        gen = _nad()
        trial = gen.generate_trial(filler_len=2)
        gt = trial.targets["grammaticality"]
        assert gt.kind == "per_trial"
        assert gt.values is True
        assert gt.mask is None

    def test_pair_index_recovery(self):
        gen = _nad()
        trial = gen.generate_trial(filler_len=2)
        idx = trial.targets["pair_index"].values
        # The dependency pair at that index should match the first/last tokens
        d1, d2 = gen.dependency_pairs[idx]
        assert trial.symbols[0] == d1
        assert trial.symbols[-1] == d2

    def test_meta_carries_paradigm_info(self):
        gen = _nad()
        trial = gen.generate_trial(filler_len=3)
        assert trial.meta["paradigm"] == "NonAdjacentDependencies"
        assert trial.meta["dependency_length"] == 3
        assert trial.meta["n_deps"] == gen.n_deps
        assert trial.meta["length"] == len(trial.symbols)

    def test_generate_trials_batch(self):
        from symseq.trial import Trial
        gen = _nad()
        trials = gen.generate_trials(n=4, filler_len=2)
        assert len(trials) == 4
        assert all(isinstance(t, Trial) for t in trials)
        assert all(len(t.symbols) == 4 for t in trials)

    def test_draw_trial_and_draw_batch_protocol(self):
        from symseq.trial import Trial
        from symseq.trial_source import TrialSource
        gen = _nad()
        assert isinstance(gen, TrialSource)
        t = gen.draw_trial()
        assert isinstance(t, Trial)
        batch = gen.draw_batch(3)
        assert len(batch) == 3 and all(isinstance(b, Trial) for b in batch)

    def test_same_seed_reproducible_trial(self):
        g1 = _nad(seed=2026)
        g2 = _nad(seed=2026)
        t1 = g1.generate_trial(filler_len=2)
        t2 = g2.generate_trial(filler_len=2)
        assert t1.symbols == t2.symbols
        assert t1.targets["pair_index"].values == t2.targets["pair_index"].values

    def test_registry_builds_nad(self):
        from symseq.generators.registry import build, registered_names
        from symseq.trial import Trial
        assert "NonAdjacentDependencies" in registered_names()
        gen = build("NonAdjacentDependencies", n_deps=3, seed=42, verbose=False)
        assert isinstance(gen, NonAdjacentDependencies)
        assert isinstance(gen.generate_trial(), Trial)
