# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Linear-algebra helpers shared across metrics and generators."""

import numpy as np


def spectral_radius(matrix: np.ndarray) -> float:
    """
    Compute the spectral radius of a square matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Square 2D matrix.

    Returns
    -------
    float
        Largest absolute eigenvalue of `matrix`.

    Notes
    -----
    For a non-negative matrix (e.g. a transition table) this equals the
    Perron-Frobenius eigenvalue. Taking the maximum over absolute values,
    rather than over the real parts, is also correct for periodic graphs
    whose spectrum contains complex eigenvalues of the same magnitude.
    """
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))
