"""
test_cfg.py

Tests for symseq.generators.cfg.CFGGenerator: PCFG-driven generation, the
grammar-as-oracle grammaticality check, negative-sample corruption, the
Trial API, and end-to-end config/registry wiring.
"""

import re

import pytest

from symseq.config import load_trial_set
from symseq.generators.cfg import CFGGenerator, NLTK_AVAILABLE
from symseq.generators.presets import cfg as cfg_presets

pytestmark = pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK not installed")


# a^n b^n with equal counts, PCFG form
ANBN = "S -> 'a' S 'b' [0.5] | 'a' 'b' [0.5]"
# plain CFG (no probabilities) -> uniformized
ANBN_CFG = "S -> 'a' S 'b' | 'a' 'b'"
# ambiguous, recursive grammar
AMBIG = "S -> S S [0.3] | 'x' S 'y' [0.3] | 'x' 'y' [0.4]"


# ----------------------------- generation -----------------------------


def test_anbn_strings_are_balanced_and_parse():
    gen = CFGGenerator(ANBN, max_length=40, seed=0)
    for _ in range(50):
        s = gen.generate_string()
        joined = "".join(s)
        assert re.fullmatch(r"a+b+", joined), f"unexpected shape: {joined}"
        assert joined.count("a") == joined.count("b")
        assert gen.is_grammatical(s)


def test_plain_cfg_is_uniformized_and_usable():
    gen = CFGGenerator(ANBN_CFG, max_length=40, seed=1)
    # Probabilities should sum to 1 per LHS.
    probs = [p.prob() for p in gen.grammar.productions()]
    assert all(0 < p <= 1 for p in probs)
    s = gen.generate_string()
    assert gen.is_grammatical(s)


def test_alphabet_is_grammar_terminals():
    gen = CFGGenerator(ANBN, seed=0)
    assert gen.alphabet == ["a", "b"]


# ----------------------------- oracle ---------------------------------


def test_is_grammatical_accepts_and_rejects():
    gen = CFGGenerator(ANBN, seed=0)
    assert gen.is_grammatical(["a", "a", "b", "b"])
    assert not gen.is_grammatical(["a", "b", "b"])  # unbalanced
    assert not gen.is_grammatical(["b", "a"])  # wrong order
    assert not gen.is_grammatical([])  # empty
    assert not gen.is_grammatical(["a", "c", "b"])  # token outside alphabet


# --------------------------- reproducibility --------------------------


def test_recursion_terminates_within_bounds():
    gen = CFGGenerator(AMBIG, max_length=30, seed=3)
    for _ in range(50):
        s = gen.generate_string()
        assert len(s) <= 30
        assert gen.is_grammatical(s)


def test_same_seed_is_reproducible():
    a = CFGGenerator(AMBIG, max_length=30, seed=7).generate_string_set(20)
    b = CFGGenerator(AMBIG, max_length=30, seed=7).generate_string_set(20)
    assert a == b


# ----------------------------- negatives ------------------------------


def test_nongrammatical_strings_fail_oracle():
    gen = CFGGenerator(ANBN, max_length=40, seed=0)
    negs = gen.generate_nongrammatical_strings(n=20, n_deviants=1)
    assert len(negs) == 20
    for s in negs:
        assert not gen.is_grammatical(s)


# ----------------------------- Trial API ------------------------------


def test_generate_trial_grammatical():
    gen = CFGGenerator(ANBN, seed=0)
    trial = gen.generate_trial(grammatical=True)
    print(trial)
    assert trial.intrinsic_targets["grammaticality"].values is True
    assert trial.intrinsic_targets["grammaticality"].granularity == "per_trial"
    assert trial.meta["paradigm"] == "CFG"
    assert trial.meta["length"] == len(trial.symbols)
    assert gen.is_grammatical(trial.symbols)


def test_generate_trial_nongrammatical():
    gen = CFGGenerator(ANBN, seed=0)
    trial = gen.generate_trial(grammatical=False)
    assert trial.intrinsic_targets["grammaticality"].values is False
    assert not gen.is_grammatical(trial.symbols)


# -------------------------- config / registry -------------------------


def test_load_trial_set_via_config():
    cfg = {
        "symseq": {
            "seed": 42,
            "generator": {
                "type": "CFG",
                "params": {"grammar": ANBN, "max_length": 30},
            },
            "trial_set": {
                "n_trials": 25,
                "splits": {"train": 20, "test": 5},
            },
        }
    }
    ts = load_trial_set(cfg)
    assert len(ts.trials) == 25
    assert ts.meta["alphabet"] == ["a", "b"]
    assert set(ts.splits) == {"train", "test"}


# ------------------------------- presets ------------------------------


@pytest.mark.parametrize("name", cfg_presets.PRESETS)
def test_preset_round_trips(name):
    """Every preset constructs, generates, and accepts its own output."""
    gen = CFGGenerator.from_preset(name, seed=0)
    assert gen.label == name
    for _ in range(30):
        s = gen.generate_string()
        assert s, "preset produced an empty string"
        assert len(s) <= gen.max_length
        assert gen.is_grammatical(s)


def test_preset_shapes():
    assert all(
        re.fullmatch(r"a+b+", "".join(CFGGenerator.from_preset("anbn", seed=i).generate_string())) for i in range(10)
    )
    # even palindrome: string equals its own reverse
    pal = CFGGenerator.from_preset("palindrome", seed=1)
    for _ in range(10):
        s = pal.generate_string()
        assert s == s[::-1]
    # dyck1: balanced parentheses
    d = CFGGenerator.from_preset("dyck1", seed=2)
    for _ in range(10):
        s = d.generate_string()
        depth = 0
        for tok in s:
            depth += 1 if tok == "(" else -1
            assert depth >= 0
        assert depth == 0


def test_from_preset_unknown_raises():
    with pytest.raises(ValueError, match="not found"):
        CFGGenerator.from_preset("does_not_exist")


def test_preset_via_config():
    cfg = {
        "symseq": {
            "seed": 7,
            "generator": {"type": "CFG", "preset": "anbn"},
            "trial_set": {"n_trials": 15},
        }
    }
    ts = load_trial_set(cfg)
    assert len(ts.trials) == 15
    assert ts.meta["alphabet"] == ["a", "b"]


test_generate_trial_grammatical()
