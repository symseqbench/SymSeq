"""Tests for topological entropy validation."""

import numpy as np
import pytest

from symseq.metrics.grammar.topological import topological_entropy


def test_lift_requires_sequence():
    with pytest.raises(ValueError, match="requires a sequence"):
        topological_entropy(method="lift", verbose=False)


def test_direct_requires_numpy_array():
    with pytest.raises(TypeError, match="Binary transition table"):
        topological_entropy(transitions=[[0, 1], [1, 0]], method="direct", verbose=False)


def test_direct_requires_square_2d_transition_table():
    transitions = np.array([[0, 1, 0], [1, 0, 1]])
    with pytest.raises(ValueError, match="square 2D"):
        topological_entropy(transitions=transitions, method="direct", verbose=False)


def test_direct_requires_binary_transition_table():
    transitions = np.array([[0.0, 0.5], [1.0, 0.0]])
    with pytest.raises(ValueError, match="binary transition table"):
        topological_entropy(transitions=transitions, method="direct", verbose=False)
