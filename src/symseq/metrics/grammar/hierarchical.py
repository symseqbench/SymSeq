# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Hierarchical structure detection via MI decay analysis."""

import numpy as np
from scipy import special, optimize
from collections import Counter
from typing import Dict, Tuple, List
from tqdm import tqdm
from joblib import Parallel, delayed


def grassberger_entropy(sequence: list[str]) -> float:
    """
    Calculate entropy using Grassberger correction for finite samples.

    Parameters
    ----------
    sequence : list of str
        Symbolic sequence.

    Returns
    -------
    float
        Grassberger-corrected entropy in bits.

    Notes
    -----
    H_hat = log2(N) - (1/N) * sum(n_i * psi(n_i))
    where psi is the digamma function.

    References
    ----------
    Grassberger, P. (1988). Finite sample corrections to entropy and
    dimension estimates. Physics Letters A, 128(6-7), 369-373.
    """
    if not sequence:
        return 0.0

    counts = np.array(list(Counter(sequence).values()))
    n_total = len(sequence)

    if len(counts) == 0 or n_total == 0:
        return 0.0

    log_n = np.log2(n_total)
    correction = np.sum(counts * special.digamma(counts)) / n_total

    return log_n - correction


def mi_decay_analysis(
    sequence: list[str],
    max_distance: int = 100,
    n_shuffles: int = 1000,
    confidence_level: float = 0.95,
    random_state: int | None = None,
    verbose: bool = False
) -> dict:
    """
    Analyze hierarchical structure via mutual information decay.

    Parameters
    ----------
    sequence : list of str
        Symbolic sequence.
    max_distance : int, default=100
        Maximum distance to analyze.
    n_shuffles : int, default=1000
        Number of shuffles for baseline correction.
    confidence_level : float, default=0.95
        Confidence level for significance testing.
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, default=False
        If True, display progress bars and intermediate information.

    Returns
    -------
    dict
        Results containing:
        - distances: Distance values
        - mi_values: Observed MI at each distance
        - mi_adjusted: Baseline-corrected MI
        - mi_baseline: Mean shuffled MI
        - mi_ci_lower/upper: Confidence intervals
        - max_significant_distance: Maximum significant distance (convergence-based)
        - max_significant_distance_ci: CI-based significant distance
        - convergence_distance: Distance where MI converges to baseline
        - convergence_threshold: Threshold used for convergence detection
        - best_model: Best-fitting decay model
        - model_fits: Fit results for all models
        - model_weights: Akaike weights for model comparison

    Notes
    -----
    Exponential decay suggests Markovian structure.
    Power-law decay indicates hierarchical organization.

    References
    ----------
    Howard-Spink, S. et al. (2024). Hierarchical structure in
    sequence learning. Nature Neuroscience.
    """
    if random_state is not None:
        np.random.seed(random_state)

    if len(sequence) < 3:
        raise ValueError("Sequence too short for MI analysis")

    max_dist = min(max_distance, len(sequence) - 1)
    distances = np.arange(1, max_dist + 1)

    if verbose:
        print(f"Computing MI decay for {len(sequence)} symbols over {len(distances)} distances...")
        print(f"Running {n_shuffles} shuffles for baseline correction...")

    mi_values = np.zeros(len(distances))
    distance_iter = tqdm(enumerate(distances), total=len(distances), desc="Computing observed MI", disable=not verbose)
    for i, d in distance_iter:
        x_seq, y_seq = _get_distance_pairs(sequence, d)
        if x_seq and y_seq:
            mi_values[i] = _mutual_information(x_seq, y_seq)

    mi_shuffled_all = np.zeros((len(distances), n_shuffles))

    shuffle_iter = tqdm(range(n_shuffles), desc="Computing shuffled baseline", disable=not verbose)
    for perm_idx in shuffle_iter:
        permuted = sequence.copy()
        np.random.shuffle(permuted)

        for i, d in enumerate(distances):
            x_seq, y_seq = _get_distance_pairs(permuted, d)
            if x_seq and y_seq:
                mi_shuffled_all[i, perm_idx] = _mutual_information(x_seq, y_seq)

    mi_baseline = np.mean(mi_shuffled_all, axis=1)
    mi_baseline_std = np.std(mi_shuffled_all, axis=1)
    mi_adjusted = mi_values - mi_baseline

    z_score = (1 - confidence_level) / 2
    z_value = -special.ndtri(z_score)
    mi_ci_lower = mi_baseline - z_value * mi_baseline_std
    mi_ci_upper = mi_baseline + z_value * mi_baseline_std

    # Find maximum significant distance using CI-based method
    # This is the LAST distance where MI is significantly above baseline
    significant = mi_values > mi_ci_upper
    max_sig_dist_ci = int(distances[significant][-1]) if np.any(significant) else 0

    # Alternative: Find where adjusted MI first drops below a threshold and stays below
    # This is more conservative and detects when correlations have decayed
    max_mi = np.max(mi_adjusted[mi_adjusted > 0]) if np.any(mi_adjusted > 0) else 0
    convergence_threshold = max(0.05 * max_mi, 0.005)  # 5% of max or 0.005 bits (stricter)
    window_size = 3  # Require 3 consecutive low points (less conservative)
    
    convergence_dist = 0
    for i in range(len(mi_adjusted) - window_size + 1):
        if np.all(mi_adjusted[i:i+window_size] < convergence_threshold):
            # Return the distance just before the convergence window
            convergence_dist = int(distances[max(0, i-1)]) if i > 0 else 0
            break
    
    # Use the more conservative (smaller) of the two methods
    # CI-based can be too liberal due to noise, convergence-based is more robust
    if convergence_dist > 0 and max_sig_dist_ci > 0:
        max_sig_dist = min(convergence_dist, max_sig_dist_ci)
    elif convergence_dist > 0:
        max_sig_dist = convergence_dist
    elif max_sig_dist_ci > 0:
        max_sig_dist = max_sig_dist_ci
    else:
        max_sig_dist = 0

    model_comparison = _compare_decay_models(distances, mi_adjusted, max_sig_dist)

    return {
        'distances': distances,
        'mi_values': mi_values,
        'mi_adjusted': mi_adjusted,
        'mi_baseline': mi_baseline,
        'mi_ci_lower': mi_ci_lower,
        'mi_ci_upper': mi_ci_upper,
        'max_significant_distance': max_sig_dist,
        'max_significant_distance_ci': max_sig_dist_ci,
        'convergence_distance': convergence_dist,
        'convergence_threshold': convergence_threshold,
        'sequence_length': len(sequence),
        'best_model': model_comparison['best_model'],
        'model_fits': model_comparison['model_fits'],
        'model_weights': model_comparison['weights'],
    }


def _compute_shuffle_mi(sequence, distances, seed):
    """Compute MI for one shuffle with fixed seed (helper for parallelization)."""
    rng = np.random.default_rng(seed)
    permuted = sequence.copy()
    rng.shuffle(permuted)
    
    mi_values = np.zeros(len(distances))
    for i, d in enumerate(distances):
        x_seq, y_seq = _get_distance_pairs(permuted, d)
        if x_seq and y_seq:
            mi_values[i] = _mutual_information(x_seq, y_seq)
    return mi_values


def mi_decay_analysis_parallel(
    sequence: list[str],
    max_distance: int = 100,
    n_shuffles: int = 1000,
    confidence_level: float = 0.95,
    random_state: int | None = None,
    n_jobs: int = -1,
    verbose: bool = False
) -> dict:
    """
    Analyze hierarchical structure via mutual information decay using parallel processing.

    Parameters
    ----------
    sequence : list of str
        Symbolic sequence.
    max_distance : int, default=100
        Maximum distance to analyze.
    n_shuffles : int, default=1000
        Number of shuffles for baseline correction.
    confidence_level : float, default=0.95
        Confidence level for significance testing.
    random_state : int, optional
        Random seed for reproducibility.
    n_jobs : int, default=-1
        Number of parallel jobs. -1 means using all processors.
    verbose : bool, default=False
        If True, display progress bars and intermediate information.

    Returns
    -------
    dict
        Results containing:
        - distances: Distance values
        - mi_values: Observed MI at each distance
        - mi_adjusted: Baseline-corrected MI
        - mi_baseline: Mean shuffled MI
        - mi_ci_lower/upper: Confidence intervals
        - max_significant_distance: Maximum significant distance (convergence-based)
        - max_significant_distance_ci: CI-based significant distance
        - convergence_distance: Distance where MI converges to baseline
        - convergence_threshold: Threshold used for convergence detection
        - best_model: Best-fitting decay model
        - model_fits: Fit results for all models
        - model_weights: Akaike weights for model comparison

    Notes
    -----
    Exponential decay suggests Markovian structure.
    Power-law decay indicates hierarchical organization.
    This parallelized version provides significant speedup for large n_shuffles.

    References
    ----------
    Howard-Spink, S. et al. (2024). Hierarchical structure in
    sequence learning. Nature Neuroscience.
    """
    if len(sequence) < 3:
        raise ValueError("Sequence too short for MI analysis")

    # Generate seeds for reproducibility
    if random_state is not None:
        base_rng = np.random.default_rng(random_state)
    else:
        base_rng = np.random.default_rng()
    
    seeds = base_rng.integers(0, 2**31, size=n_shuffles)
    
    max_dist = min(max_distance, len(sequence) - 1)
    distances = np.arange(1, max_dist + 1)

    if verbose:
        print(f"Computing MI decay for {len(sequence)} symbols over {len(distances)} distances...")
        print(f"Running {n_shuffles} shuffles for baseline correction (parallel)...")

    # Compute observed MI (sequential, fast)
    mi_values = np.zeros(len(distances))
    distance_iter = tqdm(enumerate(distances), total=len(distances), desc="Computing observed MI", disable=not verbose)
    for i, d in distance_iter:
        x_seq, y_seq = _get_distance_pairs(sequence, d)
        if x_seq and y_seq:
            mi_values[i] = _mutual_information(x_seq, y_seq)

    # Parallel shuffle computation
    mi_shuffled_list = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_compute_shuffle_mi)(sequence, distances, seed)
        for seed in tqdm(seeds, desc="Computing shuffled baseline (parallel)", disable=not verbose)
    )
    
    mi_shuffled_all = np.array(mi_shuffled_list).T  # Shape: (n_distances, n_shuffles)
    
    mi_baseline = np.mean(mi_shuffled_all, axis=1)
    mi_baseline_std = np.std(mi_shuffled_all, axis=1)
    mi_adjusted = mi_values - mi_baseline

    z_score = (1 - confidence_level) / 2
    z_value = -special.ndtri(z_score)
    mi_ci_lower = mi_baseline - z_value * mi_baseline_std
    mi_ci_upper = mi_baseline + z_value * mi_baseline_std

    # Find maximum significant distance using CI-based method
    significant = mi_values > mi_ci_upper
    max_sig_dist_ci = int(distances[significant][-1]) if np.any(significant) else 0

    # Alternative: Find where adjusted MI first drops below a threshold and stays below
    max_mi = np.max(mi_adjusted[mi_adjusted > 0]) if np.any(mi_adjusted > 0) else 0
    convergence_threshold = max(0.05 * max_mi, 0.005)
    window_size = 3
    
    convergence_dist = 0
    for i in range(len(mi_adjusted) - window_size + 1):
        if np.all(mi_adjusted[i:i+window_size] < convergence_threshold):
            convergence_dist = int(distances[max(0, i-1)]) if i > 0 else 0
            break
    
    # Use the more conservative (smaller) of the two methods
    if convergence_dist > 0 and max_sig_dist_ci > 0:
        max_sig_dist = min(convergence_dist, max_sig_dist_ci)
    elif convergence_dist > 0:
        max_sig_dist = convergence_dist
    elif max_sig_dist_ci > 0:
        max_sig_dist = max_sig_dist_ci
    else:
        max_sig_dist = 0

    model_comparison = _compare_decay_models(distances, mi_adjusted, max_sig_dist)

    return {
        'distances': distances,
        'mi_values': mi_values,
        'mi_adjusted': mi_adjusted,
        'mi_baseline': mi_baseline,
        'mi_ci_lower': mi_ci_lower,
        'mi_ci_upper': mi_ci_upper,
        'max_significant_distance': max_sig_dist,
        'max_significant_distance_ci': max_sig_dist_ci,
        'convergence_distance': convergence_dist,
        'convergence_threshold': convergence_threshold,
        'sequence_length': len(sequence),
        'best_model': model_comparison['best_model'],
        'model_fits': model_comparison['model_fits'],
        'model_weights': model_comparison['weights'],
    }


def _mutual_information(x_seq: list[str], y_seq: list[str]) -> float:
    """Calculate MI(X,Y) = H(X) + H(Y) - H(X,Y)."""
    if len(x_seq) != len(y_seq) or not x_seq:
        return 0.0

    h_x = grassberger_entropy(x_seq)
    h_y = grassberger_entropy(y_seq)

    joint_seq = [f"{x}_{y}" for x, y in zip(x_seq, y_seq)]
    h_xy = grassberger_entropy(joint_seq)

    return h_x + h_y - h_xy


def _get_distance_pairs(sequence: list[str], distance: int) -> Tuple[list, list]:
    """Get pairs of elements separated by specified distance."""
    if distance <= 0 or distance >= len(sequence):
        return [], []

    x_seq = sequence[:-distance]
    y_seq = sequence[distance:]

    return x_seq, y_seq


def _exponential_model(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Exponential decay: a * exp(-b * x)"""
    return a * np.exp(-b * x)


def _power_law_model(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Power law decay: a * x^(-b)"""
    return a * np.power(x, -b)


def _composite_model(x: np.ndarray, a: float, b: float, c: float, f: float) -> np.ndarray:
    """Composite: a * exp(-b * x) + c * x^(-f)"""
    return a * np.exp(-b * x) + c * np.power(x, -f)


def _fit_decay_model(x_data: np.ndarray, y_data: np.ndarray, model_type: str) -> dict:
    """Fit a decay model to MI data."""
    valid = (y_data > 0) & np.isfinite(y_data) & np.isfinite(x_data)
    x = x_data[valid]
    y = y_data[valid]

    if len(x) < 3:
        return {'success': False, 'aicc': np.inf}

    if model_type == 'exponential':
        model_func = _exponential_model
        p0 = [y[0], 0.1]
        bounds = ([0, 0], [np.inf, np.inf])
        n_params = 2
    elif model_type == 'power_law':
        model_func = _power_law_model
        p0 = [y[0] * x[0], 1.0]
        bounds = ([0, 0], [np.inf, np.inf])
        n_params = 2
    elif model_type == 'composite':
        model_func = _composite_model
        p0 = [y[0] * 0.5, 0.1, y[0] * 0.5, 1.0]
        bounds = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])
        n_params = 4
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    try:
        popt, _ = optimize.curve_fit(model_func, x, y, p0=p0, bounds=bounds, maxfev=5000)

        y_fitted = model_func(x, *popt)
        residuals = y - y_fitted
        mse = np.mean(residuals**2)

        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        n = len(y)
        # AICc (corrected AIC for small samples)
        aicc = n * np.log(mse) + 2 * n_params + (2 * n_params * (n_params + 1)) / (n - n_params - 1)
        # BIC (Bayesian Information Criterion - stronger penalty for complexity)
        bic = n * np.log(mse) + n_params * np.log(n)

        return {
            'success': True,
            'model_type': model_type,
            'parameters': popt,
            'aicc': aicc,
            'bic': bic,
            'r_squared': r_squared,
            'rmse': np.sqrt(mse),
        }

    except Exception:
        return {'success': False, 'model_type': model_type, 'aicc': np.inf, 'bic': np.inf}


def _compare_decay_models(distances: np.ndarray, mi_adjusted: np.ndarray, max_sig_dist: int) -> dict:
    """Fit all models and select best using BIC."""
    if max_sig_dist == 0:
        return {'best_model': 'none', 'model_fits': {}, 'weights': {}}

    # Use at least first 10 points for model fitting, or up to max_sig_dist, whichever is larger
    # This ensures we have enough data to fit models properly
    min_points_for_fitting = 10
    fit_range = max(max_sig_dist, min_points_for_fitting)
    fit_range = min(fit_range, len(distances))  # Don't exceed available data
    
    max_idx = np.where(distances <= fit_range)[0]
    x_data = distances[max_idx].astype(float)
    y_data = mi_adjusted[max_idx]

    positive = y_data > 0
    x_data = x_data[positive]
    y_data = y_data[positive]

    # Need at least 5 points to fit composite model (4 params + 1 degree of freedom)
    if len(x_data) < 5:
        return {'best_model': 'insufficient_data', 'model_fits': {}, 'weights': {}}

    model_fits = {}
    for model_type in ['exponential', 'power_law', 'composite']:
        model_fits[model_type] = _fit_decay_model(x_data, y_data, model_type)

    valid_models = {k: v for k, v in model_fits.items() if v.get('success', False)}

    if not valid_models:
        return {'best_model': 'none', 'model_fits': model_fits, 'weights': {}}

    # Use BIC (stronger penalty for complexity) instead of AICc
    best_model = min(valid_models.keys(), key=lambda k: valid_models[k]['bic'])

    min_bic = min(m['bic'] for m in valid_models.values())
    delta_bic = {k: m['bic'] - min_bic for k, m in valid_models.items()}
    # BIC weights (similar to Akaike weights but for BIC)
    exp_terms = {k: np.exp(-0.5 * delta) for k, delta in delta_bic.items()}
    sum_exp = sum(exp_terms.values())
    weights = {k: exp_val / sum_exp for k, exp_val in exp_terms.items()}

    return {
        'best_model': best_model,
        'model_fits': model_fits,
        'weights': weights
    }
