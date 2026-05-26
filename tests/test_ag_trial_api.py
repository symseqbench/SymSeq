"""
test_ag_trial_api.py

Tests for the Trial-based API of ArtificialGrammar (generate_trial / draw_trial).
The pre-existing generate_string family stays intact and is exercised via nAX tests
and downstream uses; this file focuses on the new Trial surface.
"""

import numpy as np
import pytest

from symseq.generators.ag import ArtificialGrammar


def _ag():
    return ArtificialGrammar.from_preset(preset_name="Elman", seed=42)


class TestTrialAPI:
    def test_generate_trial_returns_trial(self):
        from symseq.trial import Trial, Target
        gen = _ag()
        trial = gen.generate_trial(length_range=(3, 30))
        assert isinstance(trial, Trial)
        assert len(trial.symbols) > 0
        assert len(trial.states) == len(trial.symbols)
        assert isinstance(trial.targets["grammaticality"], Target)

    def test_symbols_and_states_are_consistent(self):
        from symseq.utils.strtools import string_as_symbols
        gen = _ag()
        trial = gen.generate_trial(length_range=(3, 30))
        # symbols is the state-list with indices stripped
        assert trial.symbols == string_as_symbols(trial.states)
        # every symbol is in the alphabet
        assert all(s in gen.alphabet for s in trial.symbols)

    def test_grammaticality_target_true_by_default(self):
        gen = _ag()
        trial = gen.generate_trial(length_range=(3, 30))
        gt = trial.targets["grammaticality"]
        assert gt.kind == "per_trial"
        assert gt.values is True
        assert gt.mask is None

    def test_grammaticality_target_false_for_nongrammatical(self):
        gen = _ag()
        trial = gen.generate_trial(length_range=(3, 30), grammatical=False, n_deviants=1)
        assert trial.targets["grammaticality"].values is False

    def test_meta_carries_paradigm_info(self):
        gen = _ag()
        trial = gen.generate_trial(length_range=(3, 30))
        assert trial.meta["paradigm"] == "ArtificialGrammar"
        assert trial.meta["label"] == gen.label
        assert trial.meta["length"] == len(trial.symbols)
        assert trial.meta["n_states"] == len(gen.states)

    def test_generate_trials_batch(self):
        from symseq.trial import Trial
        gen = _ag()
        trials = gen.generate_trials(n=4, length_range=(3, 30))
        assert len(trials) == 4
        assert all(isinstance(t, Trial) for t in trials)

    def test_draw_trial_and_draw_batch_protocol(self):
        from symseq.trial import Trial
        from symseq.trial_source import TrialSource
        gen = _ag()
        assert isinstance(gen, TrialSource)
        t = gen.draw_trial()
        assert isinstance(t, Trial)
        batch = gen.draw_batch(3)
        assert len(batch) == 3 and all(isinstance(b, Trial) for b in batch)

    def test_iter_trials_yields_trials(self):
        from itertools import islice
        from symseq.trial import Trial
        gen = _ag()
        first_five = list(islice(gen.iter_trials(length_range=(3, 30)), 5))
        assert len(first_five) == 5
        assert all(isinstance(t, Trial) for t in first_five)

    def test_same_seed_reproducible_trial(self):
        g1 = ArtificialGrammar.from_preset(preset_name="Elman", seed=2026)
        g2 = ArtificialGrammar.from_preset(preset_name="Elman", seed=2026)
        t1 = g1.generate_trial(length_range=(3, 30))
        t2 = g2.generate_trial(length_range=(3, 30))
        assert t1.symbols == t2.symbols
        assert t1.states == t2.states

    def test_registry_builds_artificial_grammar(self):
        from symseq.generators.registry import build, registered_names
        assert "ArtificialGrammar" in registered_names()
