"""Tests for optional continuous-to-symbolic transforms."""

import importlib

import numpy as np
import pytest

from symseq.symbolization import SAXSymbolizer, sax_symbolize

sax_module = importlib.import_module("symseq.symbolization.sax")


def test_sax_symbolize_wraps_saxpy_and_maps_symbols(monkeypatch):
    def fake_sax_by_chunking(
        series,
        *,
        paa_size,
        alphabet_size,
        znorm_threshold,
    ):
        assert np.array_equal(series, np.array([0.0, 1.0, 2.0, 3.0]))
        assert (paa_size, alphabet_size, znorm_threshold) == (4, 3, 0.01)
        return "abca"

    monkeypatch.setattr(sax_module, "_require_saxpy", lambda: fake_sax_by_chunking)

    assert sax_symbolize([0, 1, 2, 3], paa_size=4, alphabet_size=3) == [
        "A",
        "B",
        "C",
        "A",
    ]


def test_sax_symbolizer_preserves_recording_boundaries(monkeypatch):
    monkeypatch.setattr(
        sax_module,
        "_require_saxpy",
        lambda: lambda series, **kwargs: "ab",
    )
    symbolizer = SAXSymbolizer(paa_size=2, alphabet_size=2, symbols=("low", "high"))

    result = symbolizer.transform_many([[0.0, 1.0], [2.0, 3.0]])

    assert result == [["low", "high"], ["low", "high"]]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"paa_size": 0}, "paa_size"),
        ({"paa_size": 2, "alphabet_size": 1}, "alphabet_size"),
        ({"paa_size": 2, "alphabet_size": 3, "symbols": ("A", "B")}, "exactly 3"),
    ],
)
def test_sax_parameter_validation_happens_before_import(kwargs, message):
    with pytest.raises(ValueError, match=message):
        sax_symbolize([0.0, 1.0], **kwargs)


def test_sax_rejects_nonfinite_input(monkeypatch):
    monkeypatch.setattr(sax_module, "_require_saxpy", lambda: pytest.fail("should not import"))

    with pytest.raises(ValueError, match="finite"):
        sax_symbolize([0.0, np.nan], paa_size=2)
