# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Shannon entropy and related metrics with robust finite-sample handling."""

import numpy as np
from collections import Counter
import warnings
from typing import Union, List, Dict, Tuple, Optional


def entropy(sequence: list[str], bias_correction: bool = True) -> float:
    """
    Calculate Shannon entropy with Miller-Madow bias correction.
    
    Works well for N >= 50. For N < 50, use with caution.
    
    Parameters
    ----------
    sequence : list of str
        Full symbolic sequence.
    bias_correction : bool, default=True
        If True, applies Miller-Madow bias correction for finite sample sizes.
        Recommended for sequences with N < 500.

    Returns
    -------
    float
        Shannon entropy of the sequence (bias-corrected if enabled).

    Notes
    -----
    H(S) = -sum(P(sigma_i) * log2(P(sigma_i)))
    where P(sigma_i) is the empirical probability of token sigma_i.

    Bias Correction:
    For finite sequences, the empirical entropy H_emp systematically underestimates
    the true entropy. The Miller-Madow correction adds a first-order bias term:
    
    H_corrected = H_emp + (M - 1) / (2 * N * ln(2))
    
    where M is the number of observed symbols and N is the sequence length.
    This correction is particularly important for N < 500.
    
    Reference: Miller (1955), Paninski (2003), Lesne et al. (2009)

    Properties:
    - 0 <= H(S) <= log2(|A|) where |A| is alphabet size
    - H(S) = 0 for constant sequences
    - H(S) = log2(|A|) for uniform random sequences
    """
    if len(sequence) == 0:
        return 0.0
        
    cnt = Counter(sequence)
    N = len(sequence)
    M = len(cnt)  # Number of observed symbols
    
    # Compute empirical entropy
    H_emp = 0.0
    for count in cnt.values():
        if count > 0:
            p = count / N
            H_emp -= p * np.log2(p)
    
    # Apply Miller-Madow bias correction
    if bias_correction and N > 0:
        correction = (M - 1) / (2.0 * N * np.log(2))
        return H_emp + correction
    
    return H_emp


def block_entropy(sequence: list[str], block_size: int, bias_correction: bool = False) -> float:
    """
    Compute block entropy H_L with optional bias correction.
    
    Parameters
    ----------
    sequence : list of str
        Symbolic sequence.
    block_size : int
        Size of blocks (L-grams).
    bias_correction : bool, default=False
        Apply bias correction. WARNING: Can over-correct for large L.
        Recommend False for L > 3.
    
    Returns
    -------
    float
        Block entropy in bits.

    Notes
    -----
    H_L(S) = -sum(P(g) * log2(P(g)))
    where g are all L-gram blocks in S.

    Properties:
    - H_L(S) increases with L up to saturation
    - For i.i.d. sequences: H_L(S) = L * H(S)
    """
    n = len(sequence)
    if n < block_size:
        return 0.0

    # Create L-grams
    blocks = tuple(
        "".join(sequence[i : i + block_size]) 
        for i in range(n - block_size + 1)
    )
    
    # Use entropy() function for consistency
    return entropy(list(blocks), bias_correction=bias_correction)


def entropy_rate(sequence: list[str], 
                 max_block_size: int,
                 method: str = 'incremental',
                 min_block_size: int = 2) -> float:
    """
    Estimate entropy rate h_μ using robust convergence detection.
    
    Parameters
    ----------
    sequence : list of str
        Symbolic sequence.
    max_block_size : int
        Maximum L to consider. Should be << len(sequence).
    method : str, default='incremental'
        'incremental': Use h_k = H_{k+1} - H_k (recommended)
        'normalized': Use H_k / k (less reliable for short sequences)
    min_block_size : int, default=2
        Minimum L before checking convergence.
    
    Returns
    -------
    float
        Entropy rate estimate in bits/symbol.
    
    Notes
    -----
    For short sequences, this is the LEAST reliable metric!
    Requires N >> k^k for reliable estimation at order k.
    
    Properties:
    - 0 <= h_mu <= H(S)
    - For i.i.d. sequences: h_mu = H(S)
    - For deterministic sequences: h_mu = 0
    """
    if len(sequence) < max_block_size + 1:
        warnings.warn(
            f"Sequence too short (N={len(sequence)}) for max_block_size={max_block_size}. "
            "Entropy rate estimate will be unreliable.",
            UserWarning
        )
        max_block_size = max(2, len(sequence) // 3)
    
    if method == 'incremental':
        # Compute incremental entropies h_k = H_{k+1} - H_k
        # For stationary processes, h_k converges to h_μ (the entropy rate)
        h_incremental = []
        H_values = []
        for k in range(1, max_block_size + 1):
            H_k = block_entropy(sequence, k)
            H_values.append(H_k)
            if k > 1:
                h_k = H_k - H_values[k-2]
                h_incremental.append(h_k)
        
        if not h_incremental:
            return entropy(sequence)
        
        # Detect saturation: find where increments start dropping significantly
        # For well-sampled sequences, increments should be stable
        # For under-sampled sequences, increments drop when we run out of data
        
        if len(h_incremental) >= 4:
            # Look for where increments drop below 70% of the early average
            # Use first 2-3 increments as baseline (most reliable)
            early_increments = h_incremental[:min(3, len(h_incremental))]
            mean_early = np.mean(early_increments)
            
            saturation_idx = None
            for i in range(len(early_increments), len(h_incremental)):
                if h_incremental[i] < 0.7 * mean_early:
                    saturation_idx = i
                    break
            
            if saturation_idx is not None:
                # Saturation detected - use increments AFTER saturation (converged)
                # Early increments overestimate h_mu for sequences with memory
                converged_increments = h_incremental[saturation_idx:]
                if len(converged_increments) >= 2:
                    return np.mean(converged_increments)
                else:
                    return h_incremental[saturation_idx] if saturation_idx < len(h_incremental) else h_incremental[-1]
            
            # No saturation - check for convergence
            # Use coefficient of variation of last 3 values
            last_3 = h_incremental[-3:]
            mean_last = np.mean(last_3)
            std_last = np.std(last_3)
            
            if mean_last > 0.01:  # Only if increments are meaningful
                cv = std_last / mean_last  # Coefficient of variation
                if cv < 0.1:  # Less than 10% variation
                    return mean_last
        
        # Fallback: use mean of first half (before potential saturation)
        n_use = max(2, len(h_incremental) // 2)
        return np.mean(h_incremental[:n_use])
    
    else:  # 'normalized' method
        # Compute H_k / k for increasing k
        normalized_entropies = []
        for k in range(1, max_block_size + 1):
            H_k = block_entropy(sequence, k)
            normalized_entropies.append(H_k / k)
        
        # Similar convergence detection
        if len(normalized_entropies) >= 4:
            last_3 = normalized_entropies[-3:]
            if np.std(last_3) / np.mean(last_3) < 0.05:
                return np.mean(last_3)
        
        return np.median(normalized_entropies[-3:])


def emc(sequence: list[str], 
        max_block_size: int = 10,
        return_curve: bool = False) -> tuple:
    """
    Compute Effective Measure Complexity (Excess Entropy).
    
    EMC quantifies the mutual information between past and future,
    capturing memory and temporal dependencies.
    
    The decomposition H(1) = h_μ + E should hold, where:
    - H(1) is the Shannon entropy (block entropy at k=1)
    - h_μ is the entropy rate (asymptotic limit)
    - E is the excess entropy (EMC)
    
    Parameters
    ----------
    sequence : list of str
        Symbolic sequence.
    max_block_size : int, default=10
        Maximum L for computation. Should be chosen such that
        H_L converges (flattens out).
    return_curve : bool, default=False
        If True, return full excess curve [E(1), E(2), ..., E(L)].
    
    Returns
    -------
    tuple : (emc_value, h_mu) or (emc_value, h_mu, excess_curve)
        emc_value : float
            The excess entropy in bits
        h_mu : float
            The entropy rate in bits/symbol
        excess_curve : list of float (if return_curve=True)
            Excess values E(L) = H_L - L*h_mu for each L
    
    Notes
    -----
    Correct formula: E = Σ_{k=1}^∞ [h_k - h_μ]
    
    Equivalently: E = lim_{L→∞} [H_L - L*h_μ]
    
    The limit exists and is finite for stationary ergodic processes.
    For finite sequences, we find where H_L - L*h_μ plateaus.
    
    Properties:
    - E = 0 for memoryless (i.i.d.) sequences
    - E > 0 indicates memory/structure
    - E ≤ H(X) typically (can be higher for some processes)
    - For k-th order Markov: E ≤ k * H(X)
    
    References
    ----------
    Crutchfield & Feldman (2003), Grassberger (1986)
    """
    # First estimate entropy rate robustly
    h_mu = entropy_rate(sequence, max_block_size, method='incremental')
    
    # Compute excess curve: E(L) = H_L - L*h_mu
    excess_curve = []
    for L in range(1, max_block_size + 1):
        H_L = block_entropy(sequence, L)
        E_L = H_L - L * h_mu
        excess_curve.append(E_L)
    
    # Find where the curve plateaus (converges)
    # Strategy: Find maximum in first 2/3 of curve, then check stability
    
    if len(excess_curve) >= 4:
        # Find peak location (usually early in curve)
        search_range = max(3, len(excess_curve) * 2 // 3)
        peak_idx = np.argmax(excess_curve[:search_range])
        peak_value = excess_curve[peak_idx]
        
        # Check if subsequent values are stable near peak
        # (within 5% of peak value)
        stable_values = []
        tolerance = 0.05 * abs(peak_value) if peak_value != 0 else 0.01
        
        for i in range(peak_idx, len(excess_curve)):
            if abs(excess_curve[i] - peak_value) <= tolerance:
                stable_values.append(excess_curve[i])
            else:
                # Curve started decreasing significantly - stop
                break
        
        if len(stable_values) >= 2:
            # Found a stable plateau
            emc_value = np.mean(stable_values)
        else:
            # No plateau - use peak value (may be unreliable)
            emc_value = peak_value
            warnings.warn(
                f"EMC did not converge within max_block_size={max_block_size}. "
                "Consider increasing max_block_size or sequence length.",
                UserWarning
            )
    else:
        # Too few points - use maximum
        emc_value = max(excess_curve)
        warnings.warn(
            "Insufficient block sizes for reliable EMC. Results may be unreliable.",
            UserWarning
        )
    
    # Sanity check: EMC should be non-negative and bounded
    emc_value = max(0.0, emc_value)  # Can't be negative
    
    if return_curve:
        return emc_value, h_mu, excess_curve
    else:
        return emc_value, h_mu


def detect_saturation_point(sequence: list[str],
                            max_k: int,
                            method: str = 'adaptive') -> dict:
    """
    Detect saturation point k* where block entropy curve flattens.
    
    This is the effective memory length of the sequence. 
    Note that the method is inherently limited, particularly when dealing with small
    samples
    
    Parameters
    ----------
    sequence : list of str
        Symbolic sequence.
    max_k : int
        Maximum block size to test.
    method : str, default='adaptive'
        'adaptive': Use adaptive statistical threshold (recommended)
        'relative': Use fixed relative threshold (15% of max increment)
        'statistical': Use 2-sigma significance test
        'hybrid': Combine multiple methods
    
    Returns
    -------
    dict with keys:
        'k_star': int - Detected saturation point
        'confidence': str - 'high', 'medium', 'low'
        'method_used': str
        'increments': list - Block entropy increments
        'H_values': list - Block entropy values
        'diagnostic': dict - Additional diagnostic info
    """
    # Compute block entropies
    H_values = [block_entropy(sequence, k) for k in range(1, max_k + 1)]
    
    # Compute increments δ_k = H_{k+1} - H_k
    increments = [H_values[i+1] - H_values[i] for i in range(len(H_values) - 1)]
    
    if len(increments) < 2:
        return {
            'k_star': 1,
            'confidence': 'low',
            'method_used': method,
            'increments': increments,
            'H_values': H_values,
            'diagnostic': {'error': 'Insufficient data points'}
        }
    
    # Estimate noise level from late increments (should be small if converged)
    if len(increments) >= 4:
        noise_estimate = np.std(increments[-3:])  # Last 3 increments
    else:
        noise_estimate = np.std(increments) / 2
    
    k_star = None
    confidence = 'low'
    diagnostic = {}
    
    if method == 'statistical':
        # Method 1: Statistical significance test
        # Find where increment is not significantly different from noise
        threshold = 2 * noise_estimate  # 2-sigma
        
        for i, inc in enumerate(increments):
            if i > 0 and inc < threshold:  # Skip k=1→2 (first jump)
                k_star = i + 1  # Convert to k value
                confidence = 'high' if inc < noise_estimate else 'medium'
                break
        
        diagnostic['noise_estimate'] = noise_estimate
        diagnostic['threshold'] = threshold
    
    elif method == 'relative':
        # Method 2: Relative drop from first increment
        # Look for where increment drops significantly from maximum
        max_increment = max(increments)
        threshold_ratio = 0.15  # 15% of maximum increment
        
        for i in range(1, len(increments)):
            if increments[i] / max_increment < threshold_ratio:
                k_star = i + 1
                confidence = 'medium'
                break
        
        diagnostic['max_increment'] = max_increment
        diagnostic['threshold_ratio'] = threshold_ratio
    
    elif method == 'adaptive':
        # Method 3: Adaptive - test for constant increments (IID) vs. elbow (memory)
        #
        # Key insight: For IID sequences, increments stay roughly constant until
        # sample exhaustion. For sequences with memory, there's a clear drop after k*.
        
        if len(increments) >= 4:
            # Test if early increments are statistically constant (IID pattern)
            # Use first 3-4 increments before sample-size effects dominate
            n_early = min(4, len(increments) - 1)
            early_increments = increments[:n_early]
            
            # Coefficient of variation for early increments
            mean_early = np.mean(early_increments)
            std_early = np.std(early_increments)
            cv_early = std_early / mean_early if mean_early > 0 else 0
            
            # If CV < 15%, increments are roughly constant (likely IID or very high memory)
            # In this case, k* = 1 (no detectable memory within our resolution)
            if cv_early < 0.15:
                # Check if we're just hitting sample limits vs. true IID
                # Look at where increments start dropping significantly
                for i in range(n_early, len(increments)):
                    if increments[i] < 0.5 * mean_early:
                        # Found where sample exhaustion starts
                        # k* is just before this point
                        k_star = i
                        confidence = 'medium' if cv_early < 0.10 else 'low'
                        diagnostic['pattern'] = 'constant_increments'
                        diagnostic['cv_early'] = cv_early
                        diagnostic['sample_limit_at'] = i
                        break
                
                if k_star is None:
                    # Never dropped - truly constant
                    k_star = 1
                    confidence = 'high' if cv_early < 0.10 else 'medium'
                    diagnostic['pattern'] = 'iid_like'
                    diagnostic['cv_early'] = cv_early
            
            else:
                # Increments are NOT constant - look for elbow (memory signature)
                # Find the point where increment drops below a threshold
                
                # Use a relative threshold based on the maximum increment
                max_inc = max(increments[:n_early])  # Max from early increments
                threshold = 0.30 * max_inc  # 30% of max
                
                for i in range(1, len(increments)):
                    if increments[i] < threshold:
                        k_star = i + 1
                        
                        # Confidence based on sharpness of drop
                        if i > 0:
                            drop_ratio = increments[i] / increments[i-1]
                            if drop_ratio < 0.3:
                                confidence = 'high'
                            elif drop_ratio < 0.5:
                                confidence = 'medium'
                            else:
                                confidence = 'low'
                        else:
                            confidence = 'medium'
                        
                        diagnostic['pattern'] = 'elbow_detected'
                        diagnostic['cv_early'] = cv_early
                        diagnostic['threshold'] = threshold
                        diagnostic['drop_ratio'] = drop_ratio if i > 0 else None
                        break
        
        else:
            # Too few points
            k_star = 1
            confidence = 'low'
            diagnostic['pattern'] = 'insufficient_data'
    
    elif method == 'hybrid':
        # Method 4: Run all methods and use majority vote
        methods_to_try = ['statistical', 'relative', 'adaptive']
        k_candidates = []
        
        for m in methods_to_try:
            result = detect_saturation_point(sequence, max_k, method=m)
            k_candidates.append(result['k_star'])
        
        # Use median (robust to outliers)
        k_star = int(np.median(k_candidates))
        
        # Confidence based on agreement
        k_std = np.std(k_candidates)
        if k_std < 0.5:
            confidence = 'high'
        elif k_std < 1.5:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        diagnostic['k_candidates'] = k_candidates
        diagnostic['k_std'] = k_std
    
    # Fallback if no saturation detected
    if k_star is None:
        k_star = max_k
        confidence = 'low'
        diagnostic['warning'] = 'No clear saturation detected within max_k'
    
    # Additional validation
    N = len(sequence)
    alphabet_size = len(set(sequence))
    
    # Check if k* is reasonable given sequence length
    # Rule of thumb: Need N >> |A|^k for reliable k-gram statistics
    min_samples_per_kgram = 3
    max_reliable_k = int(np.floor(np.log(N / min_samples_per_kgram) / np.log(alphabet_size)))
    
    if k_star > max_reliable_k:
        warnings.warn(
            f"Detected k*={k_star} may be unreliable for N={N} and alphabet size={alphabet_size}. "
            f"Maximum reliable k ≈ {max_reliable_k}",
            UserWarning
        )
        confidence = 'low'
    
    diagnostic['max_reliable_k'] = max_reliable_k
    diagnostic['N'] = N
    diagnostic['alphabet_size'] = alphabet_size
    
    return {
        'k_star': k_star,
        'confidence': confidence,
        'method_used': method,
        'increments': increments,
        'H_values': H_values,
        'diagnostic': diagnostic
    }


def compute_adaptive_max_k(sequence: list[str], 
                           default_max_k: int = 6,
                           min_samples_per_kgram: int = 5) -> int:
    """
    Compute adaptive maximum block size based on sequence length and alphabet.
    
    Parameters
    ----------
    sequence : list of str
        Symbolic sequence.
    default_max_k : int, default=6
        Default maximum k if sequence is long enough.
    min_samples_per_kgram : int, default=5
        Minimum number of samples required per k-gram for reliable statistics.
    
    Returns
    -------
    int
        Adaptive maximum block size.
    
    Notes
    -----
    Uses the rule: max_k such that N >= min_samples * |A|^k
    where N is sequence length and |A| is alphabet size.
    """
    seq_len = len(sequence)
    alphabet_size = len(set(sequence))
    
    if alphabet_size <= 1:
        return min(2, seq_len - 1) if seq_len > 1 else 1
    
    # Calculate maximum k where we have enough samples
    max_k_reliable = int(np.log(seq_len / min_samples_per_kgram) / np.log(alphabet_size))
    
    # Ensure at least k=2 for meaningful analysis, but not more than default
    return max(2, min(default_max_k, max_k_reliable, seq_len - 1))


def aggregate_saturation_results(saturation_results: List[Dict]) -> Dict:
    """
    Aggregate saturation detection results across multiple sequences.
    
    Uses confidence-weighted median to determine consensus k*.
    
    Parameters
    ----------
    saturation_results : list of dict
        List of saturation detection results from detect_saturation_point().
    
    Returns
    -------
    dict with keys:
        'k_star': int - Consensus saturation point (weighted median)
        'confidence': str - Overall confidence level
        'confidence_counts': dict - Count of each confidence level
        'k_star_values': list - Individual k* values
    """
    if not saturation_results:
        return {
            'k_star': 2,
            'confidence': 'low',
            'confidence_counts': {'high': 0, 'medium': 0, 'low': 0},
            'k_star_values': []
        }
    
    k_star_values = [r['k_star'] for r in saturation_results]
    confidences = [r['confidence'] for r in saturation_results]
    
    # Weighted median using confidence levels
    conf_map = {'high': 3, 'medium': 2, 'low': 1}
    weights = [conf_map[c] for c in confidences]
    
    # Sort by k* value with corresponding weights
    sorted_pairs = sorted(zip(k_star_values, weights))
    cumsum = 0
    total = sum(weights)
    
    # Find weighted median
    k_star_consensus = sorted_pairs[-1][0]  # Default to max
    for val, weight in sorted_pairs:
        cumsum += weight
        if cumsum >= total / 2:
            k_star_consensus = val
            break
    
    # Compute confidence statistics
    high_conf_count = confidences.count('high')
    medium_conf_count = confidences.count('medium')
    low_conf_count = confidences.count('low')
    total_count = len(confidences)
    
    # Overall confidence based on proportion of high-confidence results
    if high_conf_count / total_count > 0.6:
        overall_confidence = 'high'
    elif high_conf_count / total_count > 0.3:
        overall_confidence = 'medium'
    else:
        overall_confidence = 'low'
    
    return {
        'k_star': int(k_star_consensus),
        'confidence': overall_confidence,
        'confidence_counts': {
            'high': high_conf_count,
            'medium': medium_conf_count,
            'low': low_conf_count
        },
        'k_star_values': k_star_values
    }


def _compute_block_entropies_cached(sequence: list[str], max_k: int) -> Dict[int, float]:
    """Compute all block entropies up to max_k once (helper for optimization)."""
    H_values = {}
    for k in range(1, max_k + 1):
        H_values[k] = block_entropy(sequence, k)
    return H_values


def _emc_from_cache(H_cache: Dict[int, float], max_k: int) -> Tuple[float, float]:
    """Compute EMC using pre-computed H values (helper for optimization)."""
    # Estimate entropy rate from cached values
    h_incremental = []
    for k in range(2, max_k + 1):
        h_k = H_cache[k] - H_cache[k-1]
        h_incremental.append(h_k)
    
    # Use LATE increments for h_mu (converged values, not early ones)
    # Early increments overestimate h_mu for sequences with memory
    if len(h_incremental) >= 4:
        # Find where increments stabilize (drop below 50% of early mean)
        early_mean = np.mean(h_incremental[:3])
        converged_idx = len(h_incremental)
        for i in range(3, len(h_incremental)):
            if h_incremental[i] < 0.5 * early_mean:
                converged_idx = i
                break
        # Use the last few increments (most converged)
        n_use = max(2, len(h_incremental) - converged_idx)
        h_mu = np.mean(h_incremental[-n_use:])
    elif h_incremental:
        h_mu = h_incremental[-1]  # Use last available increment
    else:
        h_mu = 0.0
    
    # Compute excess curve
    excess_curve = []
    for L in range(1, max_k + 1):
        E_L = H_cache[L] - L * h_mu
        excess_curve.append(E_L)
    
    # Find plateau (same logic as original emc function)
    if len(excess_curve) >= 4:
        search_range = max(3, len(excess_curve) * 2 // 3)
        peak_idx = np.argmax(excess_curve[:search_range])
        peak_value = excess_curve[peak_idx]
        
        stable_values = []
        tolerance = 0.05 * abs(peak_value) if peak_value != 0 else 0.01
        
        for i in range(peak_idx, len(excess_curve)):
            if abs(excess_curve[i] - peak_value) <= tolerance:
                stable_values.append(excess_curve[i])
            else:
                break
        
        if len(stable_values) >= 2:
            emc_value = np.mean(stable_values)
        else:
            emc_value = peak_value
    else:
        emc_value = max(excess_curve) if excess_curve else 0.0
    
    emc_value = max(0.0, emc_value)
    
    # Validate: h_mu + emc should approximately equal H(1)
    H_1 = H_cache[1]
    decomposition_sum = h_mu + emc_value
    if abs(decomposition_sum - H_1) > 0.5 * H_1:  # More than 50% error
        warnings.warn(
            f"Entropy decomposition inconsistent: h_μ ({h_mu:.3f}) + EMC ({emc_value:.3f}) = "
            f"{decomposition_sum:.3f} ≠ H(1) = {H_1:.3f}. "
            "Consider increasing max_block_size for better convergence.",
            UserWarning
        )

    return emc_value, h_mu


def _detect_saturation_from_cache(H_cache: Dict[int, float], max_k: int, method: str = 'adaptive') -> Dict:
    """Detect saturation using pre-computed H values (helper for optimization)."""
    H_values = [H_cache[k] for k in range(1, max_k + 1)]
    increments = [H_values[i+1] - H_values[i] for i in range(len(H_values) - 1)]
    
    # Simplified saturation detection (basic version)
    if len(increments) >= 3:
        # Find where increments drop significantly
        mean_early = np.mean(increments[:3])
        for i in range(3, len(increments)):
            if increments[i] < 0.3 * mean_early:
                return {
                    'k_star': i + 1,
                    'confidence': 'medium',
                    'method_used': method,
                    'increments': increments,
                    'H_values': H_values
                }
    
    return {
        'k_star': 2,
        'confidence': 'low',
        'method_used': method,
        'increments': increments,
        'H_values': H_values
    }


def compute_entropy_metrics_ensemble(sequences: List[list[str]],
                                     max_block_size: Optional[int] = None,
                                     min_sequence_length: int = 10,
                                     saturation_method: str = 'hybrid') -> Dict:
    """
    Compute entropy-based metrics on an ensemble of sequences.
    
    This function handles the complexity of computing block entropy, entropy rate,
    and EMC across multiple sequences with varying lengths, using adaptive max_k
    and robust aggregation.
    
    Parameters
    ----------
    sequences : list of list of str
        List of symbolic sequences.
    max_block_size : int, optional
        Maximum block size to use. If None, computed adaptively per sequence.
    min_sequence_length : int, default=10
        Minimum sequence length to include in analysis.
    saturation_method : str, default='hybrid'
        Method for saturation detection ('adaptive', 'hybrid', etc.).
    
    Returns
    -------
    dict with keys:
        'block_entropy_k2': list of float
        'entropy_rate': list of float
        'emc': list of float
        'saturation': dict from aggregate_saturation_results()
        'n_sequences_analyzed': int
        'mean_sequence_length': float
    """
    # Filter sequences by minimum length
    valid_sequences = [seq for seq in sequences if len(seq) >= min_sequence_length]
    
    if not valid_sequences:
        warnings.warn(
            f"No sequences meet minimum length requirement ({min_sequence_length}). "
            "Returning default values.",
            UserWarning
        )
        return {
            'block_entropy_k2': [],
            'entropy_rate': [],
            'emc': [],
            'saturation': aggregate_saturation_results([]),
            'n_sequences_analyzed': 0,
            'mean_sequence_length': 0.0
        }
    
    # Compute metrics for each sequence
    block_entropies_k2 = []
    entropy_rates = []
    emcs = []
    saturation_results = []
    
    for seq in valid_sequences:
        # Determine adaptive max_k for this sequence
        if max_block_size is None:
            max_k = compute_adaptive_max_k(seq, default_max_k=6)
        else:
            max_k = min(max_block_size, len(seq) - 1)
        
        # Skip if max_k is too small for meaningful analysis
        if max_k < 2:
            continue
        
        # OPTIMIZATION: Compute all block entropies once and reuse
        try:
            H_cache = _compute_block_entropies_cached(seq, max_k)
        except Exception as e:
            warnings.warn(f"Failed to compute block entropies: {e}", UserWarning)
            continue
        
        # Use cached block entropy at k=2
        try:
            block_entropies_k2.append(H_cache[2])
        except Exception as e:
            warnings.warn(f"Failed to retrieve block entropy k=2: {e}", UserWarning)
        
        # Compute entropy rate and EMC using cached values
        if max_k >= 3:
            try:
                emc_val, h_mu = _emc_from_cache(H_cache, max_k)
                entropy_rates.append(h_mu)
                emcs.append(emc_val)
            except Exception as e:
                warnings.warn(f"Failed to compute EMC/entropy rate from cache: {e}", UserWarning)
            
            # Saturation detection using cached values
            try:
                sat_result = _detect_saturation_from_cache(H_cache, max_k, method=saturation_method)
                saturation_results.append(sat_result)
            except Exception as e:
                warnings.warn(f"Failed saturation detection from cache: {e}", UserWarning)
    
    # Aggregate results
    saturation_agg = aggregate_saturation_results(saturation_results)
    
    return {
        'block_entropy_k2': block_entropies_k2,
        'entropy_rate': entropy_rates,
        'emc': emcs,
        'saturation': saturation_agg,
        'saturation_results_raw': saturation_results,  # For diagnostic plotting
        'n_sequences_analyzed': len(valid_sequences),
        'mean_sequence_length': np.mean([len(seq) for seq in valid_sequences])
    }
