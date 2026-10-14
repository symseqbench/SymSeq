# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Empirical entropy and finite-order estimates with explicit sampling diagnostics."""

from collections import Counter

import numpy as np

from symseq.utils.io import get_logger
from symseq.utils.validation import is_integer, is_real

logger = get_logger(__name__)

# Runtime ceiling for automatic order selection; fixed orders are not silently capped.
_MAX_ORDER_HARD_CAP = 30
# Average observations per observed context required by the heuristic sampling screen.
_DEFAULT_MIN_OBS_PER_CONTEXT = 5.0
# Threshold for the first-order bias approximation, not a bound on actual error.
_DEFAULT_BIAS_TOLERANCE = 0.05
# Saturation-detection methods accepted by detect_saturation_point and the ensemble
_SATURATION_METHODS = ("adaptive", "relative", "statistical", "hybrid")


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

    Blocks are tuples of symbols, so multi-character symbols never merge
    (``['AB', 'C']`` and ``['A', 'BC']`` are different blocks).

    Properties:
    - H_L(S) increases with L up to saturation
    - For i.i.d. sequences: H_L(S) = L * H(S)
    """
    n = len(sequence)
    if n < block_size:
        return 0.0

    # Create L-grams
    blocks = [tuple(sequence[i : i + block_size]) for i in range(n - block_size + 1)]

    # Use entropy() function for consistency
    return entropy(blocks, bias_correction=bias_correction)


def _validate_emc_order(order: int | str) -> None:
    if order != "bic" and not (is_integer(order) and order >= 0):
        raise ValueError(f"order must be 'bic' or a non-negative integer, got {order!r}.")


def _max_order_from_block_size(max_block_size: int) -> int:
    if not (is_integer(max_block_size) and max_block_size >= 1):
        raise ValueError(f"max_block_size must be a positive integer, got {max_block_size!r}.")
    return int(max_block_size) - 1


def _empirical_entropy(counts: Counter, total: int) -> float:
    p = np.fromiter(counts.values(), dtype=float) / total
    return float(-np.sum(p * np.log2(p)))


def _block_stats(sequence: list[str], max_block_size: int) -> tuple[list[float], list[float], list[int], list[int]]:
    """
    Block statistics for L = 0..max_block_size, in bits.

    Returns ``(H, h_cond, K, n)``, each indexed by block length L:

    - ``H[L]``: plug-in block entropy, as `block_entropy` without bias correction.
    - ``h_cond[L]``: plug-in conditional entropy of the last symbol of an L-block given its first
      L - 1 symbols, from the L-blocks alone. It equals H_L - H_{L-1} up to
      finite-window edge terms, but is never negative and is exactly 0 for a deterministic continuation.
    - ``K[L]``: number of distinct L-blocks observed.
    - ``n[L]``: number of L-block windows, ``len(sequence) - L + 1``.

    L = 0 is the empty block: H_0 = 0, one distinct block, ``len(sequence)`` windows; h_cond[0]
    is undefined (NaN).
    """
    N = len(sequence)
    H, h_cond, K, n = [0.0], [np.nan], [1], [N]
    for L in range(1, max_block_size + 1):
        total = N - L + 1
        counts = Counter(tuple(sequence[i : i + L]) for i in range(total))
        prefixes = Counter()
        for block, count in counts.items():
            prefixes[block[:-1]] += count
        H.append(_empirical_entropy(counts, total))
        h_cond.append(max(0.0, H[-1] - _empirical_entropy(prefixes, total)))
        K.append(len(counts))
        n.append(total)
    return H, h_cond, K, n


def _bic_scores(sequence: list[str], max_order: int) -> dict[int, float]:
    """
    BIC score of the order-R Markov fit for R = 0..max_order (lower is better).

    Every order is scored on the same ``len(sequence) - max_order`` positions (those with at least
    `max_order` predecessors), so the likelihoods are comparable across orders.
    """
    alphabet_size = len(set(sequence))
    n_obs = len(sequence) - max_order
    windows = Counter(tuple(sequence[i : i + max_order + 1]) for i in range(n_obs))
    scores = {}
    for R in range(max_order + 1):
        blocks, contexts = Counter(), Counter()
        for window, count in windows.items():
            block = window[max_order - R :]  # the scored symbol and its R predecessors
            blocks[block] += count
            contexts[block[:-1]] += count
        h_bits = _empirical_entropy(blocks, n_obs) - _empirical_entropy(contexts, n_obs)
        neg2_loglik = 2.0 * np.log(2) * n_obs * max(0.0, h_bits)
        n_parameters = len(contexts) * (alphabet_size - 1)
        scores[R] = float(neg2_loglik + n_parameters * np.log(n_obs))
    return scores


def _validate_emc_settings(order, max_order, bias_tolerance, min_obs_per_context) -> None:
    """Validate estimator settings before computing or filtering any sequences."""
    _validate_emc_order(order)
    if max_order is not None and not (is_integer(max_order) and max_order >= 0):
        raise ValueError("max_order must be None or a non-negative integer.")
    for name, value, allow_zero in (
        ("bias_tolerance", bias_tolerance, True),
        ("min_obs_per_context", min_obs_per_context, False),
    ):
        if not is_real(value) or not np.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
            bound = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be finite and {bound}.")


def _sampling_order_limit(sequence: list[str], cap: int, minimum: float) -> int:
    """Return the last order in the contiguous range passing the sampling screen.

    Contexts are prefixes of observed (R+1)-blocks, so each has a following symbol.
    Stop at the first failing order. Return -1 if even order zero fails. This is a
    heuristic screening policy, not a bound on estimable memory or independent data.
    """
    for R in range(cap + 1):
        n = len(sequence) - R
        contexts = {tuple(sequence[i : i + R]) for i in range(n)}
        if n / len(contexts) < minimum:
            return R - 1
    return cap


def _estimate_emc(
    sequence: list[str],
    order: int | str,
    max_order: int | None,
    bias_tolerance: float,
    min_obs_per_context: float,
    extra_blocks: int = 0,
) -> dict:
    """Compute the shared result, allowing unavailable fixed orders in ensemble records.

    Extra block statistics support curves and saturation independently of the EMC
    sampling screen. Public callers validate arguments before entering this core.
    """
    N = len(sequence)
    fixed = order != "bic"
    requested_cap = _MAX_ORDER_HARD_CAP if max_order is None else int(max_order)
    cap = min(requested_cap, _MAX_ORDER_HARD_CAP, N - 2)
    # Fixed orders bypass search caps but still receive a sampling diagnostic.
    screen_cap = min(int(order), N - 2) if fixed else cap
    sampling_limit = _sampling_order_limit(sequence, screen_cap, min_obs_per_context)
    search_limit = None if fixed else sampling_limit
    reasons = []
    if not fixed:
        if cap == _MAX_ORDER_HARD_CAP:
            reasons.append("runtime_cap")
        if max_order is not None and cap == max_order:
            reasons.append("caller_cap")
        if cap == N - 2:
            reasons.append("sequence_length")
        if sampling_limit < cap:
            reasons.append("sampling_screen")

    scores = None
    R = None
    status = "computed"
    warnings = []
    if fixed:
        if order > N - 2:
            status = "insufficient_data"
            warnings.append("fixed_order_unavailable")
        else:
            R = int(order)
            if sampling_limit < R:
                status = "sparse"
                warnings.append("sampling_screen_failed")
    elif sampling_limit < 1 and not (max_order == 0 and sampling_limit == 0):
        status = "insufficient_data"
        warnings.append("insufficient_order_comparison")
    else:
        scores = _bic_scores(sequence, sampling_limit)
        R = min(scores, key=scores.__getitem__)

    needed = (R + 1) if fixed and R is not None else max(1, sampling_limit + 1)
    H, h_cond, K, n = _block_stats(sequence, min(N, max(needed, extra_blocks, 1)))
    result = {
        "entropy": float(H[1]),
        "entropy_method": "empirical",
        "bias_correction": False,
        "entropy_rate": None,
        "emc": None,
        "order": R,
        "requested_order": int(order) if fixed else "bic",
        "order_method": "fixed" if fixed else "bic",
        "bic_variant": None if fixed else "observed_contexts",
        "bic_scores": scores,
        "n_symbols": N,
        "max_order_requested": None if max_order is None else int(max_order),
        "max_order_searched": sampling_limit if scores is not None else None,
        "search_order_limit": search_limit,
        "sampling_order_limit": sampling_limit,
        "search_limit_reasons": reasons,
        "n_scored": N - sampling_limit if scores is not None else None,
        "order_at_search_limit": sampling_limit == R if scores is not None else None,
        "min_obs_per_context": float(min_obs_per_context),
        "emc_bias_estimate": None,
        "entropy_rate_bias_estimate": None,
        "bias_ok": None,
        "bias_tolerance": float(bias_tolerance),
        "status": status,
        "warnings": warnings,
        "limitations": [
            "sampling_uncertainty_not_quantified",
            "dependence_not_assessed",
            "truncation_not_assessed",
            "asymptotic_convergence_not_established",
        ],
        "units": {"entropy": "bits", "entropy_rate": "bits/symbol", "emc": "bits"},
        "block_entropies": H[1:],
        "conditional_entropies": h_cond[1:],
    }
    if R is None:
        return result

    rate = float(h_cond[R + 1])
    raw_emc = float(H[R] - R * rate)

    def empirical_bias(L):
        return -(K[L] - 1) / (2.0 * n[L] * np.log(2))

    # Match the prefix support and denominator to the conditional-entropy sample.
    # A deterministic continuation then has zero conditional bias approximation.
    prefix_count = len({tuple(sequence[i : i + R]) for i in range(n[R + 1])})
    rate_bias = float(-(K[R + 1] - prefix_count) / (2.0 * n[R + 1] * np.log(2)))
    emc_bias = float(empirical_bias(R) - R * rate_bias)
    bias_ok = abs(emc_bias) <= bias_tolerance and abs(rate_bias) <= bias_tolerance
    result.update(
        entropy_rate=rate,
        emc=max(0.0, raw_emc),
        entropy_rate_bias_estimate=rate_bias,
        emc_bias_estimate=emc_bias,
        bias_ok=bool(bias_ok),
    )
    if not bias_ok:
        warnings.append("bias_approximation_above_tolerance")
    if raw_emc < -1e-12:
        warnings.append("negative_emc_clamped")
    return result


def _warn_emc(result: dict) -> None:
    """Log actionable calculation diagnostics without implying statistical confidence."""
    if result["warnings"]:
        logger.warning(
            "EMC status=%s, order=%s: %s. See estimate_emc diagnostics; "
            "uncertainty and longer dependencies are not assessed.",
            result["status"],
            result["order"],
            ", ".join(result["warnings"]),
        )


def _number_or_nan(value) -> float:
    """Map an unavailable structured value to a numerical NaN."""
    return float("nan") if value is None else float(value)


def estimate_emc(
    sequence: list[str],
    order: int | str = "bic",
    max_order: int | None = None,
    bias_tolerance: float = _DEFAULT_BIAS_TOLERANCE,
    min_obs_per_context: float = _DEFAULT_MIN_OBS_PER_CONTEXT,
) -> dict:
    """
    Estimate finite-order entropy rate and excess entropy with explicit diagnostics.

    Parameters
    ----------
    sequence : list of str
        At least two symbols from a source assumed stationary. Trials are not pooled.
    order : int or 'bic', default='bic'
        Fixed non-negative Markov order, or automatic observed-context BIC selection.
        A fixed order is never lowered and must permit at least two block windows.
    max_order : int or None, default=None
        Automatic search cap. None uses the runtime ceiling of 30. Explicit caps
        above 30 are capped at 30 and reported; zero is preserved. Ignored for fixed
        orders. The sequence length and sampling screen can restrict the search further.
    bias_tolerance : float, default=0.05
        Non-negative finite threshold for the absolute first-order bias approximation:
        bits for EMC and bits/symbol for the rate. It is not an actual error bound.
    min_obs_per_context : float, default=5
        Positive finite minimum average number of observations per observed context.
        Automatic selection stops at the first order failing this heuristic screen.

    Returns
    -------
    dict
        Calculation results containing the following entries:

        ``entropy`` : float
            Uncorrected empirical single-symbol entropy, in bits.
        ``entropy_method`` : {'empirical'}
            Method used to estimate ``entropy``.
        ``bias_correction`` : bool
            Whether a bias correction was applied to ``entropy``. Always ``False``.
        ``entropy_rate`` : float or None
            Order-R entropy-rate estimate, in bits per symbol, or ``None`` when
            unavailable.
        ``emc`` : float or None
            Order-R excess-entropy estimate, in bits, or ``None`` when unavailable.
        ``order`` : int or None
            Markov order used for the estimates, or ``None`` when unavailable.
        ``requested_order`` : int or {'bic'}
            Fixed order requested by the caller or ``'bic'`` for automatic selection.
        ``order_method`` : {'fixed', 'bic'}
            Method used to determine ``order``.
        ``bic_variant`` : {'observed_contexts'} or None
            BIC parameter-count variant, or ``None`` for a fixed order.
        ``bic_scores`` : dict[int, float] or None
            BIC score for each candidate order, or ``None`` when no search ran.
        ``n_symbols`` : int
            Number of symbols in the input sequence.
        ``max_order_requested`` : int or None
            Automatic search cap requested by the caller, or ``None`` when omitted.
        ``max_order_searched`` : int or None
            Highest order included in a completed BIC search, or ``None`` when no
            search ran.
        ``search_order_limit`` : int or None
            Screened automatic-search limit, or ``None`` for a fixed order.
        ``sampling_order_limit`` : int
            Highest order in the contiguous range passing the sampling screen. A
            value of -1 means that order zero failed.
        ``search_limit_reasons`` : list[str]
            Codes identifying constraints on the automatic search range.
        ``n_scored`` : int or None
            Common number of observations used for BIC scores, or ``None`` when no
            search ran.
        ``order_at_search_limit`` : bool or None
            Whether BIC selected the highest searched order, or ``None`` when no
            search ran. This is not evidence by itself of missing memory.
        ``min_obs_per_context`` : float
            Minimum average observations per observed context used by the sampling
            screen.
        ``emc_bias_estimate`` : float or None
            First-order bias approximation for ``emc``, in bits, or ``None`` when
            EMC is unavailable. The approximation is not subtracted.
        ``entropy_rate_bias_estimate`` : float or None
            First-order bias approximation for ``entropy_rate``, in bits per symbol,
            or ``None`` when the rate is unavailable. The approximation is not
            subtracted.
        ``bias_ok`` : bool or None
            Whether both absolute bias approximations meet ``bias_tolerance``, or
            ``None`` when the estimates are unavailable.
        ``bias_tolerance`` : float
            Threshold applied to the absolute bias approximations.
        ``status`` : {'computed', 'sparse', 'insufficient_data'}
            Calculation outcome. ``'computed'`` does not establish accuracy or
            convergence.
        ``warnings`` : list[str]
            Codes for calculation diagnostics encountered by the estimator.
        ``limitations`` : list[str]
            Codes for uncertainty, dependence, truncation, and convergence properties
            that the estimator does not assess.
        ``units`` : dict[str, str]
            Units for ``entropy``, ``entropy_rate``, and ``emc``.
        ``block_entropies`` : list[float]
            Empirical block entropies H_L for block lengths 1 through the computed
            limit, in bits.
        ``conditional_entropies`` : list[float]
            Empirical conditional entropies H(X_L | X_1..X_(L-1)) for block lengths
            1 through the computed limit, in bits per symbol.

    Raises
    ------
    ValueError
        If arguments are invalid, fewer than two symbols are supplied, or a fixed
        order needs blocks longer than len(sequence) - 1.

    Notes
    -----
    The shared estimator computes h_R = H(X_(R+1) | X_1..X_R) from matched block
    and prefix counts, then E_R = max(0, H_R - R*h_R). These equal asymptotic rate
    and EMC at the population level for stationary sources of order at most R.
    Empirical values have sampling error; larger orders can add both bias and variance.

    BIC scores all orders on N-M common positions, where M is the screened cap,
    with penalty K_R*(|A|-1)*log(N-M) for observed contexts K_R. This is a variant
    of standard BIC, whose unrestricted dimension is |A|**R*(|A|-1); its consistency
    theorem is not claimed here. The selected order can differ across conditions.
    Prefer a declared fixed resolution for comparisons when scientifically appropriate.
    An explicit max_order=0 permits the caller's order-zero-only candidate set;
    otherwise automatic selection needs at least orders zero and one to pass the screen.

    The Miller-Madow bias approximation ignores temporal dependence, variance,
    selection uncertainty and truncation, and can fail even when bias_ok is True.
    Some hidden-state sources have infinite observed Markov order. Longer memory
    cannot be ruled out by any of these diagnostics. Formal language class alone
    does not determine whether excess entropy is finite. No bootstrap is performed.

    References
    ----------
    Crutchfield and Feldman (2003), doi:10.1063/1.1530990.
    Paninski (2003), doi:10.1162/089976603321780272.
    Csiszar and Shields (2000), doi:10.1214/aos/1015957472 (standard BIC).
    """
    _validate_emc_settings(order, max_order, bias_tolerance, min_obs_per_context)
    if len(sequence) < 2:
        raise ValueError("At least two symbols are needed to estimate an entropy rate.")
    if order != "bic" and order > len(sequence) - 2:
        raise ValueError(
            f"order={order} needs longer blocks; this sequence supports at most order {len(sequence) - 2}."
        )
    return _estimate_emc(sequence, order, max_order, bias_tolerance, min_obs_per_context)


def entropy_rate(
    sequence: list[str],
    max_block_size: int,
    method: str = "incremental",
    min_block_size: int = 2,
    order: int | str = "bic",
    *,
    bias_tolerance: float = _DEFAULT_BIAS_TOLERANCE,
    min_obs_per_context: float = _DEFAULT_MIN_OBS_PER_CONTEXT,
) -> float:
    """
    Return a finite-order entropy-rate estimate, or NaN when unavailable.

    Parameters
    ----------
    sequence : list of str
        At least two symbols.
    max_block_size : int
        Positive block-length cap. Incremental selection searches orders up to L-1,
        subject to the sampling screen and runtime ceiling of `estimate_emc`.
    method : {'incremental', 'normalized'}, default='incremental'
        Incremental uses `estimate_emc`; normalized retains the H_L/L heuristic.
        Normalized results do not establish convergence and carry no bias diagnostic.
    min_block_size : int, default=2
        Unused; retained for backward compatibility.
    order : int or 'bic', default='bic'
        Order for the incremental estimator. Ignored by the normalized heuristic.
    bias_tolerance : float, default=0.05
        Threshold for the approximate bias check, as in `estimate_emc`.
    min_obs_per_context : float, default=5
        Sampling screen, as in `estimate_emc`.

    Returns
    -------
    float
        Bits/symbol. Use `estimate_emc` for settings, status and limitations.

    Raises
    ------
    ValueError
        If the method, settings or sequence length are invalid.
    """
    if method not in ("incremental", "normalized"):
        raise ValueError("method must be 'incremental' or 'normalized'.")
    cap = _max_order_from_block_size(max_block_size)
    _validate_emc_settings(order, cap, bias_tolerance, min_obs_per_context)
    if len(sequence) < 2:
        raise ValueError("At least two symbols are needed to estimate an entropy rate.")
    if method == "incremental":
        result = estimate_emc(sequence, order, cap, bias_tolerance, min_obs_per_context)
        _warn_emc(result)
        return _number_or_nan(result["entropy_rate"])

    max_k = min(max_block_size, len(sequence) - 1)
    values = [block_entropy(sequence, k) / k for k in range(1, max_k + 1)]
    if len(values) >= 4:
        last = values[-3:]
        mean = np.mean(last)
        if mean == 0 or np.std(last) / mean < 0.05:
            return float(mean)
    return float(np.median(values[-3:]))


def emc(
    sequence: list[str],
    max_block_size: int | None = None,
    return_curve: bool = False,
    order: int | str = "bic",
    *,
    bias_tolerance: float = _DEFAULT_BIAS_TOLERANCE,
    min_obs_per_context: float = _DEFAULT_MIN_OBS_PER_CONTEXT,
) -> tuple:
    """
    Return finite-order EMC and entropy rate using the shared estimator.

    Parameters
    ----------
    sequence : list of str
        At least two symbols.
    max_block_size : int or None, default=None
        BIC order cap L-1, or the automatic sampling limit and runtime ceiling if None.
        An explicit value also requests that many curve points (capped at N-1).
    return_curve : bool, default=False
        Return H_L - L*h_R as a descriptive curve, not a convergence test.
    order : int or 'bic', default='bic'
        Fixed or selected order as in `estimate_emc`.
    bias_tolerance : float, default=0.05
        Threshold for the approximate bias check.
    min_obs_per_context : float, default=5
        Average observations per observed context required by the sampling screen.

    Returns
    -------
    emc_value : float
        Excess-entropy estimate, in bits, or NaN when unavailable.
    entropy_rate_value : float
        Entropy-rate estimate, in bits per symbol, or NaN when unavailable.
    excess_curve : list[float], optional
        Descriptive values H_L - L*h_R for each computed block length. Returned only
        when ``return_curve`` is ``True`` and does not establish convergence.

    Raises
    ------
    ValueError
        If settings, sequence length or a requested fixed order are invalid.
    """
    cap = None if max_block_size is None else _max_order_from_block_size(max_block_size)
    result = estimate_emc(sequence, order, cap, bias_tolerance, min_obs_per_context)
    _warn_emc(result)
    E, rate = _number_or_nan(result["emc"]), _number_or_nan(result["entropy_rate"])
    if return_curve:
        H = result["block_entropies"]
        if max_block_size is not None:
            count = min(max_block_size, len(sequence) - 1)
            if count > len(H):
                H = H + [block_entropy(sequence, L) for L in range(len(H) + 1, count + 1)]
            H = H[:count]
        return E, rate, [value - L * rate for L, value in enumerate(H, start=1)]
    return E, rate


def detect_saturation_point(sequence: list[str], max_k: int, method: str = "adaptive") -> dict:
    """
    Detect saturation point k* where block entropy curve flattens.

    This describes empirical curve flattening only. Undersampling can produce the
    same shape as genuine predictability. It neither identifies the source's memory
    length nor establishes entropy-rate or EMC convergence, and does not select the
    order used by the EMC estimator.

    Parameters
    ----------
    sequence : list of str
        Symbolic sequence.
    max_k : int
        Maximum block size to test.
    method : str, default='adaptive'
        'adaptive': Use adaptive statistical threshold (recommended)
        'relative': Use fixed relative threshold (15% of max increment)
        'statistical': Use a heuristic two-standard-deviation threshold (not a significance test)
        'hybrid': Combine multiple methods

    Returns
    -------
    dict
        Detection results containing the following entries:

        ``k_star`` : int
            Detected empirical saturation point.
        ``confidence`` : {'high', 'medium', 'low'}
            Heuristic confidence label, not a calibrated probability.
        ``method_used`` : {'adaptive', 'relative', 'statistical', 'hybrid'}
            Saturation-detection method used.
        ``interpretation`` : {'empirical_flattening'}
            Interpretation assigned to the detected pattern.
        ``confidence_calibrated`` : bool
            Whether ``confidence`` is statistically calibrated. Always ``False``.
        ``increments`` : list[float]
            Successive empirical block-entropy increments, in bits.
        ``H_values`` : list[float]
            Empirical block entropies for block lengths 1 through ``max_k``, in bits.
        ``diagnostic`` : dict[str, object]
            Method-dependent diagnostic values and sampling information.

    Raises
    ------
    ValueError
        If `method` is not one of the listed methods.

    Notes
    -----
    An observed constant sequence returns k* = 1; this does not establish the source has no memory.
    """
    _validate_saturation_method(method)
    H_values = [block_entropy(sequence, k) for k in range(1, max_k + 1)]
    return _detect_saturation_from_entropies(H_values, len(sequence), len(set(sequence)), method)


def _validate_saturation_method(method: str) -> None:
    if method not in _SATURATION_METHODS:
        raise ValueError(f"Unknown saturation method {method!r}. Choose one of {_SATURATION_METHODS}.")


def _detect_saturation_from_entropies(
    H_values: list[float], N: int, alphabet_size: int, method: str, warn_unreliable: bool = True
) -> dict:
    """
    Detect the saturation point from block entropies H_1..H_max_k (shared core).

    Used by `detect_saturation_point` and by the ensemble path, so both apply the same rule.
    If `warn_unreliable` is False, the log message for an unreliable k* is suppressed (the result is
    still marked as low confidence).
    """
    _validate_saturation_method(method)
    max_k = len(H_values)

    # Compute increments δ_k = H_{k+1} - H_k
    increments = [H_values[i + 1] - H_values[i] for i in range(len(H_values) - 1)]

    if len(increments) < 2:
        return {
            "k_star": 1,
            "confidence": "low",
            "method_used": method,
            "interpretation": "empirical_flattening",
            "confidence_calibrated": False,
            "increments": increments,
            "H_values": H_values,
            "diagnostic": {"error": "Insufficient data points"},
        }

    if alphabet_size <= 1:
        # The observed constant sequence has a flat empirical entropy curve.
        return {
            "k_star": 1,
            "confidence": "high",
            "method_used": method,
            "interpretation": "empirical_flattening",
            "confidence_calibrated": False,
            "increments": increments,
            "H_values": H_values,
            "diagnostic": {"pattern": "constant_sequence", "N": N, "alphabet_size": alphabet_size},
        }

    # Descriptive variation in the tail, not a standard error or convergence test.
    noise_estimate = np.std(increments[-3:]) if len(increments) >= 4 else np.std(increments) / 2

    k_star = None
    confidence = "low"
    diagnostic = {}

    if method == "statistical":
        # Method 1: Heuristic threshold on empirical increments, not a significance test
        threshold = 2 * noise_estimate  # 2-sigma

        for i, inc in enumerate(increments):
            if i > 0 and inc < threshold:  # Skip k=1→2 (first jump)
                k_star = i + 1  # Convert to k value
                confidence = "high" if inc < noise_estimate else "medium"
                break

        diagnostic["noise_estimate"] = noise_estimate
        diagnostic["threshold"] = threshold

    elif method == "relative":
        # Method 2: Relative drop from first increment
        # Look for where increment drops significantly from maximum
        max_increment = max(increments)
        threshold_ratio = 0.15  # 15% of maximum increment

        for i in range(1, len(increments)):
            if increments[i] / max_increment < threshold_ratio:
                k_star = i + 1
                confidence = "medium"
                break

        diagnostic["max_increment"] = max_increment
        diagnostic["threshold_ratio"] = threshold_ratio

    elif method == "adaptive":
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
                        confidence = "medium" if cv_early < 0.10 else "low"
                        diagnostic["pattern"] = "constant_increments"
                        diagnostic["cv_early"] = cv_early
                        diagnostic["sample_limit_at"] = i
                        break

                if k_star is None:
                    # Never dropped - truly constant
                    k_star = 1
                    confidence = "high" if cv_early < 0.10 else "medium"
                    diagnostic["pattern"] = "iid_like"
                    diagnostic["cv_early"] = cv_early

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
                            drop_ratio = increments[i] / increments[i - 1]
                            if drop_ratio < 0.3:
                                confidence = "high"
                            elif drop_ratio < 0.5:
                                confidence = "medium"
                            else:
                                confidence = "low"
                        else:
                            confidence = "medium"

                        diagnostic["pattern"] = "elbow_detected"
                        diagnostic["cv_early"] = cv_early
                        diagnostic["threshold"] = threshold
                        diagnostic["drop_ratio"] = drop_ratio if i > 0 else None
                        break

        else:
            # Too few points
            k_star = 1
            confidence = "low"
            diagnostic["pattern"] = "insufficient_data"

    elif method == "hybrid":
        # Method 4: Run all methods and use majority vote
        methods_to_try = ["statistical", "relative", "adaptive"]
        k_candidates = []

        for m in methods_to_try:
            # reliability is checked once below
            result = _detect_saturation_from_entropies(H_values, N, alphabet_size, m, warn_unreliable=False)
            k_candidates.append(result["k_star"])

        # Use median (robust to outliers)
        k_star = int(np.median(k_candidates))

        # Confidence based on agreement
        k_std = np.std(k_candidates)
        if k_std < 0.5:
            confidence = "high"
        elif k_std < 1.5:
            confidence = "medium"
        else:
            confidence = "low"

        diagnostic["k_candidates"] = k_candidates
        diagnostic["k_std"] = k_std

    # Fallback if no saturation detected
    if k_star is None:
        k_star = max_k
        confidence = "low"
        diagnostic["warning"] = "No clear saturation detected within max_k"

    # Additional validation
    # Check if k* is reasonable given sequence length
    # Rule of thumb: Need N >> |A|^k for reliable k-gram statistics
    min_samples_per_kgram = 3
    max_reliable_k = int(np.floor(np.log(N / min_samples_per_kgram) / np.log(alphabet_size)))

    if k_star > max_reliable_k:
        if warn_unreliable:
            logger.warning(
                f"Detected k*={k_star} may be unreliable for N={N} and alphabet size={alphabet_size}. "
                f"Maximum reliable k ≈ {max_reliable_k}"
            )
        confidence = "low"

    diagnostic["max_reliable_k"] = max_reliable_k
    diagnostic["N"] = N
    diagnostic["alphabet_size"] = alphabet_size

    return {
        "k_star": k_star,
        "confidence": confidence,
        "method_used": method,
        "interpretation": "empirical_flattening",
        "confidence_calibrated": False,
        "increments": increments,
        "H_values": H_values,
        "diagnostic": diagnostic,
    }


def compute_adaptive_max_k(sequence: list[str], default_max_k: int = 6, min_samples_per_kgram: int = 5) -> int:
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


def aggregate_saturation_results(saturation_results: list[dict]) -> dict:
    """
    Aggregate saturation detection results across multiple sequences.

    Uses a heuristic-label-weighted median to summarize empirical flattening.
    These labels are not calibrated statistical confidence or evidence of convergence.

    Parameters
    ----------
    saturation_results : list of dict
        List of saturation detection results from detect_saturation_point().

    Returns
    -------
    dict
        Aggregated detection results containing the following entries:

        ``interpretation`` : {'empirical_flattening'}
            Interpretation assigned to the aggregated result.
        ``confidence_calibrated`` : bool
            Whether ``confidence`` is statistically calibrated. Always ``False``.
        ``n_detections`` : int
            Number of saturation results included in the aggregation.
        ``k_star`` : int
            Heuristic-label-weighted median saturation point, or 2 when no results
            are supplied.
        ``confidence`` : {'high', 'medium', 'low'}
            Overall heuristic confidence label.
        ``confidence_counts`` : dict[str, int]
            Numbers of ``'high'``, ``'medium'``, and ``'low'`` input labels.
        ``k_star_values`` : list[int]
            Saturation points from the individual input results.
    """
    if not saturation_results:
        return {
            "interpretation": "empirical_flattening",
            "confidence_calibrated": False,
            "n_detections": 0,
            "k_star": 2,
            "confidence": "low",
            "confidence_counts": {"high": 0, "medium": 0, "low": 0},
            "k_star_values": [],
        }

    k_star_values = [r["k_star"] for r in saturation_results]
    confidences = [r["confidence"] for r in saturation_results]

    # Weighted median using confidence levels
    conf_map = {"high": 3, "medium": 2, "low": 1}
    weights = [conf_map[c] for c in confidences]

    # Sort by k* value with corresponding weights
    sorted_pairs = sorted(zip(k_star_values, weights, strict=True))
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
    high_conf_count = confidences.count("high")
    medium_conf_count = confidences.count("medium")
    low_conf_count = confidences.count("low")
    total_count = len(confidences)

    # Overall confidence based on proportion of high-confidence results
    if high_conf_count / total_count > 0.6:
        overall_confidence = "high"
    elif high_conf_count / total_count > 0.3:
        overall_confidence = "medium"
    else:
        overall_confidence = "low"

    return {
        "interpretation": "empirical_flattening",
        "confidence_calibrated": False,
        "n_detections": len(saturation_results),
        "k_star": int(k_star_consensus),
        "confidence": overall_confidence,
        "confidence_counts": {"high": high_conf_count, "medium": medium_conf_count, "low": low_conf_count},
        "k_star_values": k_star_values,
    }


def compute_entropy_metrics_ensemble(
    sequences: list[list[str]],
    max_block_size: int | None = None,
    min_sequence_length: int = 10,
    saturation_method: str = "hybrid",
    emc_order: int | str = "bic",
    bias_tolerance: float = _DEFAULT_BIAS_TOLERANCE,
    min_obs_per_context: float = _DEFAULT_MIN_OBS_PER_CONTEXT,
) -> dict:
    """
    Compute per-sequence empirical entropy, finite-order EMC and descriptive saturation.

    Parameters
    ----------
    sequences : list of list of str
        Independent input sequences; never concatenated or pooled.
    max_block_size : int or None, default=None
        EMC search cap L-1, screened as in `estimate_emc`. For saturation this is the
        requested block limit; None uses `compute_adaptive_max_k` for that diagnostic.
    min_sequence_length : int, default=10
        Minimum included length, at least two. Input indices are returned after filtering.
    saturation_method : str, default='hybrid'
        Descriptive flattening heuristic, as in `detect_saturation_point`. It does not
        select the EMC order or establish memory length or convergence.
    emc_order : int or 'bic', default='bic'
        Fixed or selected order. A fixed order unavailable for an individual sequence
        produces an insufficient_data record without dropping that sequence.
    bias_tolerance : float, default=0.05
        Threshold for the approximate bias check, not an actual error bound.
    min_obs_per_context : float, default=5
        Sampling screen as in `estimate_emc`.

    Returns
    -------
    dict
        Ensemble results containing the following entries. Per-sequence lists are
        aligned with ``records`` and ``input_indices`` unless stated otherwise.

        ``entropy`` : list[float]
            Empirical single-symbol entropy for each included sequence, in bits.
        ``block_entropy_k2`` : list[float]
            Empirical order-two block entropy for each included sequence, in bits.
        ``entropy_rate`` : list[float]
            Entropy-rate estimate for each included sequence, in bits per symbol.
            Unavailable estimates are NaN.
        ``emc`` : list[float]
            Excess-entropy estimate for each included sequence, in bits. Unavailable
            estimates are NaN.
        ``emc_order`` : list[float]
            Markov order used for each included sequence, represented numerically.
            Unavailable orders are NaN.
        ``emc_bias_estimate`` : list[float]
            First-order EMC bias approximation for each included sequence, in bits.
            Unavailable estimates are NaN.
        ``entropy_rate_bias_estimate`` : list[float]
            First-order entropy-rate bias approximation for each included sequence,
            in bits per symbol. Unavailable estimates are NaN.
        ``records`` : list[dict]
            Full ``estimate_emc`` results for each included sequence, extended with
            ``input_index``, ``block_entropy_k2``, and ``saturation``.
        ``input_indices`` : list[int]
            Indices of included sequences in the original input.
        ``emc_bias_ok`` : list[bool or None]
            Bias-threshold result for each included sequence, or ``None`` when the
            bias approximations are unavailable.
        ``emc_status`` : list[str]
            EMC calculation status for each included sequence.
        ``emc_order_method`` : {'fixed', 'bic'}
            Method used to determine EMC orders.
        ``saturation`` : dict
            Aggregation of all successful descriptive saturation detections.
        ``saturation_results_raw`` : list[dict]
            Successful saturation detections. Unlike the other per-sequence lists,
            this list omits sequences for which saturation was unavailable.
        ``n_sequences_analyzed`` : int
            Number of sequences meeting ``min_sequence_length``, including sequences
            for which EMC was unavailable.
        ``mean_sequence_length`` : float
            Mean length of included sequences, or 0.0 when none were included.

    Raises
    ------
    ValueError
        If any global setting is invalid, including for an empty ensemble.

    Notes
    -----
    Computed status and bias_ok do not establish accuracy. Uncertainty, dependence,
    longer memory and asymptotic convergence are not assessed. See record limitations.
    """
    _validate_saturation_method(saturation_method)
    cap = None if max_block_size is None else _max_order_from_block_size(max_block_size)
    _validate_emc_settings(emc_order, cap, bias_tolerance, min_obs_per_context)
    if not is_integer(min_sequence_length) or min_sequence_length < 2:
        raise ValueError("min_sequence_length must be an integer of at least two.")
    records = []
    saturation_results = []
    for index, seq in enumerate(sequences):
        if len(seq) < min_sequence_length:
            continue
        max_k = (
            compute_adaptive_max_k(seq, default_max_k=6)
            if max_block_size is None
            else min(max_block_size, len(seq) - 1)
        )
        est = _estimate_emc(seq, emc_order, cap, bias_tolerance, min_obs_per_context, extra_blocks=max(2, max_k))
        H = est["block_entropies"]
        sat = None
        if max_k >= 3:
            sat = _detect_saturation_from_entropies(H[:max_k], len(seq), len(set(seq)), saturation_method)
            saturation_results.append(sat)
        est.update(input_index=index, block_entropy_k2=H[1], saturation=sat)
        records.append(est)

    fields = {
        "entropy": "entropy",
        "block_entropy_k2": "block_entropy_k2",
        "entropy_rate": "entropy_rate",
        "emc": "emc",
        "emc_order": "order",
        "emc_bias_estimate": "emc_bias_estimate",
        "entropy_rate_bias_estimate": "entropy_rate_bias_estimate",
    }
    result = {key: [_number_or_nan(r[field]) for r in records] for key, field in fields.items()}
    result.update(
        records=records,
        input_indices=[r["input_index"] for r in records],
        emc_bias_ok=[r["bias_ok"] for r in records],
        emc_status=[r["status"] for r in records],
        emc_order_method="bic" if emc_order == "bic" else "fixed",
        saturation=aggregate_saturation_results(saturation_results),
        saturation_results_raw=saturation_results,
        n_sequences_analyzed=len(records),
        mean_sequence_length=float(np.mean([r["n_symbols"] for r in records])) if records else 0.0,
    )
    diagnostics = Counter(reason for r in records for reason in r["warnings"])
    if diagnostics:
        logger.warning("Ensemble EMC diagnostics: %s. See records for statuses and limitations.", dict(diagnostics))
    return result
