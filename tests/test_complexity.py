"""
test_complexity.py

Tests for the complexity metrics.

Author: Barna Zajzon
"""

from statistics import mean
from turtle import st
from weakref import ref
import pytest
import numpy as np

# local imports
from symseq.grammars.ag import ArtificialGrammar
from symseq.grammars import ag_presets
from symseq.metrics import complexity


ref_string_sets = {
    "Meulemans97": {
        "Experiment 1A": {
            "train": [
                "MXRVXT",
                "VMTRRRR",
                "MXTRRR",
                "VXVRMXT",
                "VXVRVM",
                "VMRVVVV",
                "MXRTMVR",
                "VMRMXTR",
                "MXR",
                "VMRMVRV",
                "MVRVM",
                "VMRMVRM",
                "VMRMVXR",
                "MXRTVXT",
                "MXRMVXR",
                "MVXTR",
            ],
            "test": {
                "GA": [
                    "MVXR",
                    "MVXRMXR",
                    "MXRVMRV",
                    "MXRVVM",
                    "MXT",
                    "MXTR",
                    "VMRTMXR",
                    "VMRTVXT",
                ],
                "GNA": [
                    "MVXTRRX",
                    "VMTRRRX",
                    "VXTRRX",
                    "VXTRX",
                    "VXVT",
                    "VXVTR",
                    "VXVTRX",
                    "VXVTX",
                ],
                "NGA": [
                    "MVRMXR",
                    "MXRMXRR",
                    "MXRVMXR",
                    "VMRMTR",
                    "VMRR",
                    "VMRT",
                    "VMV",
                    "VMVMXT",
                ],
                "NGNA": [
                    "MTR",
                    "MTXRMTR",
                    "MVTRVW",
                    "MVXRMTR",
                    "MVXTT",
                    "VMTRRRT",
                    "VRTX",
                    "VXVRTXT",
                ],
            },
        }
    }
}


def xtest_associative_chunk_strength():
    train_string_set = ["VXM", "MSVRXM", "MSV", "MVRXRRM", "VXVRXSV", "VXSSV", "VXSSSVS", "VXV", "MSSVRXR", "MSSVRXV"]
    train_string_set = [list(s) for s in train_string_set]
    # test_string_set = ["VXRRM"]
    # test_string_set = ["VXRR"]
    # test_string_set = ["VXVS"]
    test_string_set = ["VXRRRRM"]
    # test_string_set = ["SVSM"]
    test_string_set = [list(s) for s in test_string_set]

    strengths = acs_bailey2008(train_string_set, test_string_set, n_range=(2, 3))
    print(strengths)

    raise

    grammar = ArtificialGrammar(rng=np.random.default_rng(42), **ag_presets.FSG_G3)
    train_string_set, train_grammaticality = grammar.generate_string_set(n_strings=3, nongramm_fraction=0.0)
    test_string_set, test_grammaticality = grammar.generate_string_set(n_strings=2, nongramm_fraction=0.5)
    strengths = acs_bailey2008(train_string_set, test_string_set, n=2)
    assert len(strengths) == 2
    assert np.isclose(strengths[0], 0.1818, atol=0.01)

    # test with an unknown bigram
    test_string_set.append(["A", "B"])
    strengths = acs_bailey2008(train_string_set, test_string_set, n=2)
    assert len(strengths) == 3
    assert strengths[-1] == 0.0

    train_string_set = [["A", "B", "A"]]
    test_string_set = [["A", "B", "A"]]
    strengths = acs_bailey2008(train_string_set, test_string_set, n=2)
    assert len(strengths) == 1
    assert np.isclose(np.mean(strengths), 0.5)

    train_string_set = [["A", "B", "A", "B", "A"]]
    test_string_set = [["A", "B", "A"]]
    strengths = acs_bailey2008(train_string_set, test_string_set, n=2)
    assert len(strengths) == 1
    assert np.isclose(np.mean(strengths), 0.5)

    # dataset with high chunk strengths
    train_string_set = [["A", "B"]] * 1000 + [[str(i), str(i + 1)] for i in range(1, 101)]
    test_string_set = [["A", "B"]]
    strengths = acs_bailey2008(train_string_set, test_string_set, n=2)
    assert len(strengths) == 1
    assert strengths[0] > 0.95

    # def test_mean_acs_knowlton96():
    train_set = [
        "MXRVXT",
        "VMTRRRR",
        "MXTRRR",
        "VXVRMXT",
        "VXVRVM",
        "VMRVVVV",
        "MXRTMVR",
        "VMRMXTR",
        "MXR",
        "VMRMVRV",
        "MVRVM",
        "VMRMVRM",
        "VMRMVXR",
        "MXRTVXT",
        "MXRMVXR",
        "MVXTR",
    ]

    g_s = [
        "MXRMXT",
        "VXTRRR",
        "VXVRW",
        "MXRTMXR",
        "MVR",
        "MXRVM",
        "VMRMVXT",
        "MXRMVXT",
    ]
    g_ns = [
        "MVXRV",
        "VXVR",
        "MXRTVMR",
        "MVXRVMT",
        "MXRMVRM",
        "MVRVWM",
        "VXTRRRX",
        "VXTRRX",
    ]

    ng_s = [
        "VMTRRRT",
        "VXVRTXT",
        "VMRVWR",
        "VMRTXTR",
        "VMRMTRV",
        "MMRMVRM",
        "MXRTRXT",
        "MVXTT",
    ]
    ng_ns = [
        "MVXRMTR",
        "MXRMTRV",
        "VMM",
        "MVXRWR",
        "VMRVMM",
        "VXVRMTR",
        "VXVTRRM",
        "VMRTTXT",
    ]

    train_set = [list(s) for s in train_set]
    g_s = [list(s) for s in g_s]
    g_ns = [list(s) for s in g_ns]

    # mean_acs = complexity.gacs_knowlton96(train_set, [g_s[0]], n_range=(2, 3))
    # print(mean_acs)
    mean_acs = complexity.global_acs_knowlton96(train_set, g_s, n_range=(2, 3))
    # print(mean_acs)
    # print(np.mean(mean_acs))
    # raise
    # mean_acs = complexity.gacs_knowlton96(train_set, ng_s + ng_ns, n_range=(2, 3))
    print(np.mean(mean_acs))
    raise


def test_gacs_knowlton96__meulemans97_1A():
    dataset = ref_string_sets["Meulemans97"]["Experiment 1A"]

    train_set = [list(s) for s in dataset["train"]]

    test = list(["MXRVVM"])
    mean_acs = complexity.global_acs_knowlton96(train_set, test, n_range=(2, 3))
    assert np.isclose(mean_acs[0], 4.22, atol=0.01)

    ga = [list(s) for s in dataset["test"]["GA"]]
    mean_acs = complexity.global_acs_knowlton96(train_set, ga, n_range=(2, 3))
    assert np.isclose(np.mean(mean_acs), 4.63, atol=0.01)


def test_anchor_acs_knowlton96__meulemans97_1A():
    dataset = ref_string_sets["Meulemans97"]["Experiment 1A"]

    train_set = [list(s) for s in dataset["train"]]

    test = list(["MXRVVM"])
    mean_acs = complexity.anchor_acs_knowlton96(train_set, test, n_range=(2, 3))
    assert np.isclose(mean_acs[0], 3.25, atol=0.01)

    ga = [list(s) for s in dataset["test"]["GA"]]
    mean_acs = complexity.anchor_acs_knowlton96(train_set, ga, n_range=(2, 3))
    assert np.isclose(np.mean(mean_acs), 2.9, atol=0.1)

    nga = [list(s) for s in dataset["test"]["NGA"]]
    mean_acs = complexity.mean_anchor_acs_knowlton96(train_set, nga, n_range=(2, 3))
    print(mean_acs)
    assert np.isclose(mean_acs, 2.75, atol=0.01)

    # train_set = [list(s) for s in ["ABCE", "ABCDEF", "ACDE"]]
    # test = [list(s) for s in ["ABCD"]]
    # mean_acs = complexity.anchor_acs_knowlton96(train_set, test, n_range=(2, 3))
    # print(mean_acs)
