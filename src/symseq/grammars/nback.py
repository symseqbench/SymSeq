# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""
n-back task with controlled match and lure structure.

Generates symbolic sequences for n-back working-memory experiments. Unlike the
classical AGL grammars (e.g., nAX, NAD), n-back has no formal grammar describing
"valid" streams — every symbol drawn from the alphabet is admissible. Instead,
this generator uses a position-targeted stochastic procedure:

1. Reserve positions for true n-back matches (target rate `p_match`).
2. Reserve positions for lures at user-specified lags (e.g., (n-1)- or (n+1)-back).
3. Fill remaining positions by sampling from the alphabet under per-position
   constraints (e.g., avoid accidental n-back matches).
4. Verify the realised match/lure counts equal the intended counts; retry the
   sequence on conflict.

This produces sequences with exact (not stochastic) class balance, suitable for
both human cognitive experiments and ML training datasets.

Notes
-----
Future work: optional frequency-balanced alphabet (each symbol appears ~L/V times)
for Sternberg-style controls.
"""

from __future__ import annotations

import numpy as np

from symseq.core.sequencer import SymbolicSequencer
from symseq.utils.io import get_logger

logger = get_logger(__name__)


class NBack(SymbolicSequencer):
    """
    Controlled n-back symbolic sequence generator.

    Each generated sequence has length `seq_length`. A position i (with i >= n)
    is a "match" iff seq[i] == seq[i-n]. With `match_count_mode="round"`
    (default), the realised match count over the eligible window of size
    (L - n) is exactly round(p_match * (L - n)).

    Optional lures place items that match a non-n lag (e.g., (n-1)- or
    (n+1)-back) to probe interference. Designated lure positions are
    guaranteed not to also satisfy the true n-back rule.

    Parameters
    ----------
    label : str
        Sequencer label. Default "n-back".
    n : int
        n-back distance (>= 1). Default 2.
    seq_length : int
        Sequence length L. Must be > n. Default 30.
    alphabet : list of str, optional
        Explicit alphabet. Takes precedence over `alphabet_size`.
    alphabet_size : int, optional
        Size of auto-generated alphabet (symbols "0", "1", ...). Default 8.
    p_match : float
        Target match rate, computed over the eligible window of size (L - n).
        Default 0.3.
    lure_offsets : tuple of int
        Lag deltas relative to n. E.g., (-1, +1) inserts (n-1)- and
        (n+1)-back lures. Each offset k must satisfy n + k >= 1 and k != 0.
        Default ().
    p_lure : float or dict[int, float]
        Lure rate per offset, computed over each offset's eligible window
        (size L - max(n, n+k)). Scalar shares the rate across all offsets;
        dict maps offset -> rate. Default 0.0.
    avoid_accidental_matches : bool
        If True, filler positions never accidentally satisfy seq[i] == seq[i-n].
        Default True.
    avoid_accidental_lures : bool
        If True, filler positions never accidentally satisfy
        seq[i] == seq[i-(n+k)] for any configured lure offset k. Default False.
    match_count_mode : {"round", "sample"}
        How to compute the per-sequence match count from p_match. "round" is
        deterministic and yields exact rates per sequence; "sample" draws from
        Binomial(L-n, p_match). Default "round".
    max_attempts : int
        Maximum sequence regeneration attempts before raising RuntimeError.
        Default 100.
    rng : numpy.random.Generator, optional
        RNG to use. If None and `seed` is provided, built from `seed`. If both
        are None, parent class warns about non-reproducibility.
    seed : int, optional
        Seed used to build RNG when `rng` is None.
    verbose : bool
        Verbose initialization logging. Default False.
    """

    def __init__(
        self,
        label: str = "n-back",
        n: int = 2,
        seq_length: int = 30,
        alphabet: list[str] | None = None,
        alphabet_size: int | None = 8,
        p_match: float = 0.3,
        lure_offsets: tuple[int, ...] = (),
        p_lure: float | dict[int, float] = 0.0,
        avoid_accidental_matches: bool = True,
        avoid_accidental_lures: bool = False,
        match_count_mode: str = "round",
        max_attempts: int = 100,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
        verbose: bool = False,
    ):
        if rng is None and seed is not None:
            rng = np.random.default_rng(seed)

        if alphabet is not None:
            super().__init__(label=label, alphabet=alphabet, rng=rng, verbose=verbose)
        else:
            super().__init__(
                label=label, alphabet_size=alphabet_size, rng=rng, verbose=verbose,
            )

        self.n = n
        self.seq_length = seq_length
        self.p_match = float(p_match)
        self.lure_offsets = tuple(int(k) for k in lure_offsets)
        self._p_lure_input = p_lure
        self.avoid_accidental_matches = bool(avoid_accidental_matches)
        self.avoid_accidental_lures = bool(avoid_accidental_lures)
        self.match_count_mode = match_count_mode
        self.max_attempts = int(max_attempts)
        self.seed = seed
        self.verbose = bool(verbose)

        self._validate_params()
        self._setup_lure_rates()
        self._validate_seq_length(self.seq_length)
        self._intended_n_match, self._intended_n_lure = self._compute_intended_counts(
            self.seq_length
        )

        # Cache alphabet as numpy object array for faster .choice
        self._alphabet_arr = np.array(self.alphabet, dtype=object)

    # ============================== Validation ==============================

    def _validate_params(self) -> None:
        """Validate L-independent parameters."""
        if not isinstance(self.n, int) or self.n < 1:
            raise ValueError(f"n must be a positive integer, got {self.n!r}.")
        if not (0.0 <= self.p_match <= 1.0):
            raise ValueError(f"p_match must be in [0, 1], got {self.p_match}.")
        if self.match_count_mode not in {"round", "sample"}:
            raise ValueError(
                f"match_count_mode must be 'round' or 'sample', got {self.match_count_mode!r}."
            )
        for k in self.lure_offsets:
            if k == 0:
                raise ValueError("lure_offsets must not contain 0 (that's a true match).")
            if self.n + k <= 0:
                raise ValueError(
                    f"Invalid lure offset {k}: n + k must be >= 1 (n={self.n})."
                )
        if len(set(self.lure_offsets)) != len(self.lure_offsets):
            raise ValueError(f"lure_offsets must be unique, got {self.lure_offsets}.")
        if self.alphabet_size < 2 and (
            self.avoid_accidental_matches or self.lure_offsets
        ):
            raise ValueError(
                "alphabet_size must be >= 2 when avoid_accidental_matches=True "
                "or any lure is configured."
            )
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}.")

    def _validate_seq_length(self, L: int) -> None:
        """Validate the L-dependent constraints (called from __init__ and per
        call when a `seq_length` override is supplied).
        """
        if not isinstance(L, (int, np.integer)) or L <= self.n:
            raise ValueError(
                f"seq_length must be an integer > n (got seq_length={L!r}, n={self.n})."
            )
        for k in self.lure_offsets:
            if max(self.n, self.n + k) >= L:
                raise ValueError(
                    f"Lure offset {k} has empty eligible window "
                    f"(seq_length={L}, n={self.n})."
                )

    def _setup_lure_rates(self) -> None:
        """Coerce p_lure to a dict {offset: rate} and validate."""
        p = self._p_lure_input
        if isinstance(p, dict):
            extra = set(p) - set(self.lure_offsets)
            missing = set(self.lure_offsets) - set(p)
            if extra:
                raise ValueError(
                    f"p_lure has keys not in lure_offsets: {sorted(extra)}."
                )
            if missing:
                raise ValueError(
                    f"p_lure missing entries for offsets: {sorted(missing)}."
                )
            rates = {int(k): float(v) for k, v in p.items()}
        else:
            rates = {int(k): float(p) for k in self.lure_offsets}
        for k, v in rates.items():
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"p_lure[{k}] must be in [0, 1], got {v}.")
        self.p_lure: dict[int, float] = rates

    def _compute_intended_counts(self, L: int) -> tuple[int, dict[int, int]]:
        """Compute intended match/lure counts for a given sequence length and
        verify the total fits the eligible window.

        Returns
        -------
        (n_match, n_lure_per_offset)
        """
        n = self.n
        n_match = int(round(self.p_match * (L - n)))
        n_lure = {
            k: int(round(self.p_lure[k] * (L - max(n, n + k))))
            for k in self.lure_offsets
        }
        total = n_match + sum(n_lure.values())
        if total > L - n:
            raise ValueError(
                f"Cannot fit {n_match} matches and {sum(n_lure.values())} lures "
                f"into {L - n} eligible positions (seq_length={L}, n={n})."
            )
        return n_match, n_lure

    # ============================== Allocation ==============================

    def _allocate_positions(
        self, L: int, intended_n_match: int, intended_n_lure: dict[int, int]
    ) -> dict:
        """Choose match, lure, and filler positions for one sequence of length L.

        Parameters
        ----------
        L : int
            Sequence length.
        intended_n_match : int
            Intended match count (used when `match_count_mode == "round"`).
        intended_n_lure : dict[int, int]
            Intended lure counts per offset (used when `match_count_mode == "round"`).
        """
        n = self.n

        if self.match_count_mode == "round":
            n_match = intended_n_match
        else:
            n_match = int(self.rng.binomial(L - n, self.p_match))

        eligible = np.arange(n, L)
        match_idx = self.rng.choice(eligible, size=n_match, replace=False)
        used = set(int(i) for i in match_idx)

        lure_idx: dict[int, np.ndarray] = {}
        for k in self.lure_offsets:
            window_start = max(n, n + k)
            available = np.array(
                [i for i in range(window_start, L) if i not in used], dtype=int
            )
            if self.match_count_mode == "round":
                n_lure_k = intended_n_lure[k]
            else:
                n_lure_k = int(self.rng.binomial(L - window_start, self.p_lure[k]))
            # When the lure lag is 1 (n+k == 1), two consecutive lures at i, i+1
            # deterministically force a true n-back match at i+1, which would
            # violate the "lure is not also a match" guarantee. Enforce min
            # spacing of 2 between such lures.
            if n + k == 1:
                chosen = self._sample_with_min_spacing(
                    available, n_lure_k, min_distance=2,
                )
                if chosen is None:
                    raise RuntimeError(
                        f"Cannot place {n_lure_k} lag-1 lures (offset {k}) "
                        f"with min spacing 2 in window [{window_start}, {L})."
                    )
            else:
                if n_lure_k > len(available):
                    raise RuntimeError(
                        f"Cannot place {n_lure_k} lures at offset {k}: "
                        f"only {len(available)} positions free."
                    )
                chosen = self.rng.choice(available, size=n_lure_k, replace=False)
            lure_idx[k] = chosen
            used.update(int(i) for i in chosen)

        burn_in = np.arange(0, n)
        all_filled = used | set(int(i) for i in burn_in)
        filler = np.array(
            [i for i in range(L) if i not in all_filled], dtype=int
        )

        role: list = [None] * L
        for i in burn_in:
            role[int(i)] = "burn_in"
        for i in match_idx:
            role[int(i)] = "match"
        for k, idxs in lure_idx.items():
            for i in idxs:
                role[int(i)] = ("lure", int(k))
        for i in filler:
            role[int(i)] = "filler"

        return {
            "match": np.sort(match_idx).astype(int),
            "lure": {k: np.sort(v).astype(int) for k, v in lure_idx.items()},
            "filler": filler,
            "burn_in": burn_in,
            "role": role,
        }

    def _sample_with_min_spacing(
        self, available: np.ndarray, k: int, min_distance: int
    ) -> np.ndarray | None:
        """Greedily pick k positions from `available` with pairwise distance
        >= `min_distance`. Returns None if not possible. Sampling order is
        random (controlled by self.rng).
        """
        if k == 0:
            return np.array([], dtype=int)
        order = self.rng.permutation(len(available))
        chosen: list[int] = []
        for idx in order:
            pos = int(available[idx])
            if all(abs(pos - c) >= min_distance for c in chosen):
                chosen.append(pos)
                if len(chosen) == k:
                    return np.array(sorted(chosen), dtype=int)
        return None

    # ================================ Filling ================================

    def _build_extra_forbid(self, alloc: dict) -> dict[int, list[int]]:
        """Map each fill position to earlier-position indices whose value must
        differ at fill time, so a designated lure position does not also
        satisfy the true n-back rule.
        """
        n = self.n
        extra: dict[int, list[int]] = {}
        for k, idxs in alloc["lure"].items():
            for i in idxs.tolist():
                p_lure_src = i - (n + k)
                p_match_src = i - n
                later = max(p_lure_src, p_match_src)
                earlier = min(p_lure_src, p_match_src)
                extra.setdefault(later, []).append(earlier)
        return extra

    def _fill(self, L: int, alloc: dict) -> list[str] | None:
        """Fill a sequence given allocation. Returns the sequence on success,
        or None if a position has no admissible value (caller should retry).
        """
        n = self.n
        seq: list[str | None] = [None] * L
        extra_forbid = self._build_extra_forbid(alloc)
        roles = alloc["role"]

        for i in range(L):
            role = roles[i]
            if role == "burn_in":
                seq[i] = str(self.rng.choice(self._alphabet_arr))
            elif role == "match":
                seq[i] = seq[i - n]
            elif isinstance(role, tuple) and role[0] == "lure":
                k = role[1]
                seq[i] = seq[i - (n + k)]
            else:  # "filler"
                forbidden: set[str] = set()
                if self.avoid_accidental_matches and i >= n:
                    forbidden.add(seq[i - n])  # type: ignore[arg-type]
                if self.avoid_accidental_lures:
                    for k in self.lure_offsets:
                        src = i - (n + k)
                        if 0 <= src < i:
                            forbidden.add(seq[src])  # type: ignore[arg-type]
                for src in extra_forbid.get(i, []):
                    if 0 <= src < i and seq[src] is not None:
                        forbidden.add(seq[src])  # type: ignore[arg-type]
                allowed = [s for s in self.alphabet if s not in forbidden]
                if not allowed:
                    return None
                seq[i] = str(self.rng.choice(np.array(allowed, dtype=object)))

        return seq  # type: ignore[return-value]

    # ============================== Generation ==============================

    def generate_string(
        self, seq_length: int | None = None, *args, **kwargs
    ) -> list[str]:
        """Generate one n-back sequence.

        Parameters
        ----------
        seq_length : int, optional
            Override the instance-level `self.seq_length` for this call only.
            The instance is not mutated. If None (default), uses
            `self.seq_length` and the precomputed intended counts.

        Returns
        -------
        list of str
            Sequence of length L drawn from the alphabet.

        Raises
        ------
        ValueError
            If `seq_length` violates L-dependent validation (too small, empty
            lure window, capacity overflow).
        RuntimeError
            If `max_attempts` regenerations all fail to produce a valid sequence.
        """
        if seq_length is None:
            L = self.seq_length
            intended_n_match = self._intended_n_match
            intended_n_lure = self._intended_n_lure
        else:
            self._validate_seq_length(seq_length)
            L = int(seq_length)
            intended_n_match, intended_n_lure = self._compute_intended_counts(L)

        last_report = None
        for _ in range(self.max_attempts):
            alloc = self._allocate_positions(L, intended_n_match, intended_n_lure)
            seq = self._fill(L, alloc)
            if seq is None:
                continue
            report = self._verify(seq, alloc)
            if report["ok"]:
                return seq
            last_report = report
        raise RuntimeError(
            f"Failed to generate a valid n-back sequence after "
            f"{self.max_attempts} attempts. Last report: {last_report}. "
            f"Consider increasing alphabet_size or relaxing constraints."
        )

    def generate_string_set(
        self, n_samples: int = 1, seq_length: int | None = None, **kwargs
    ) -> list[list[str]]:
        """Generate a batch of n-back sequences (all sharing the same length).

        Parameters
        ----------
        n_samples : int
            Number of sequences to generate. Default 1.
        seq_length : int, optional
            Override the instance-level `self.seq_length` for this batch.

        Returns
        -------
        list of list of str
            List of n_samples sequences, each of length `seq_length` (or
            `self.seq_length` if not given).
        """
        return [
            self.generate_string(seq_length=seq_length, **kwargs)
            for _ in range(n_samples)
        ]

    # ============================== Labeling ================================

    def label_sequence(self, seq: list[str]) -> np.ndarray:
        """Per-position binary labels.

        Returns
        -------
        np.ndarray of int8, shape (L,)
            -1 for burn-in positions [0, n); 1 for n-back matches; 0 otherwise.
        """
        L, n = len(seq), self.n
        labels = np.zeros(L, dtype=np.int8)
        labels[:n] = -1
        for i in range(n, L):
            labels[i] = 1 if seq[i] == seq[i - n] else 0
        return labels

    def label_sequence_multiclass(self, seq: list[str]) -> np.ndarray:
        """Per-position multi-class labels.

        Codes
        -----
        -1   burn-in (i < n)
         0   no match of any kind
         1   true n-back match
         2.. lure of type lure_offsets[code-2] (in user-given order)

        Notes
        -----
        True matches always win over coincidental lures. If multiple lure
        offsets coincide at one position, the first in `lure_offsets` wins.
        """
        L, n = len(seq), self.n
        labels = np.zeros(L, dtype=np.int8)
        labels[:n] = -1
        for i in range(n, L):
            if seq[i] == seq[i - n]:
                labels[i] = 1
                continue
            assigned = False
            for offset_idx, k in enumerate(self.lure_offsets):
                src = i - (n + k)
                if 0 <= src < L and seq[i] == seq[src]:
                    labels[i] = 2 + offset_idx
                    assigned = True
                    break
            if not assigned:
                labels[i] = 0
        return labels

    def label_trial(self, seq: list[str]) -> dict:
        """Summary report for one sequence, derived from the sequence alone.

        Returns
        -------
        dict
            {
                "n_match_realised": int,
                "n_lure_realised": dict[int, int],
                "match_rate": float,
                "length": int,
            }

        Notes
        -----
        Without allocation metadata, this cannot distinguish intended from
        accidental occurrences. Use `_verify` internally for that.
        """
        L = len(seq)
        labels = self.label_sequence(seq)
        n_match = int((labels == 1).sum())
        n_lure: dict[int, int] = {}
        for k in self.lure_offsets:
            count = 0
            for i in range(max(self.n, self.n + k), L):
                if labels[i] == 1:
                    continue
                if seq[i] == seq[i - (self.n + k)]:
                    count += 1
            n_lure[k] = count
        return {
            "n_match_realised": n_match,
            "n_lure_realised": n_lure,
            "match_rate": n_match / max(1, L - self.n),
            "length": L,
        }

    def verify_sequence(self, seq: list[str]) -> dict:
        """Public alias for `label_trial`."""
        return self.label_trial(seq)

    def _verify(self, seq: list[str], alloc: dict) -> dict:
        """Compare realised sequence with the intended allocation."""
        n = self.n
        L = len(seq)
        labels = self.label_sequence(seq)

        intended_match_set = set(int(i) for i in alloc["match"])
        realised_match_set = {i for i in range(n, L) if labels[i] == 1}
        n_accidental_match = len(realised_match_set - intended_match_set)
        n_missing_match = len(intended_match_set - realised_match_set)

        intended_lure_sets = {
            k: set(int(i) for i in alloc["lure"][k]) for k in self.lure_offsets
        }
        accidental_lure: dict[int, int] = {}
        missing_lure: dict[int, int] = {}
        n_lure_realised: dict[int, int] = {}
        for k in self.lure_offsets:
            realised = set()
            for i in range(max(n, n + k), L):
                if labels[i] == 1:
                    continue
                if seq[i] == seq[i - (n + k)]:
                    realised.add(i)
            n_lure_realised[k] = len(realised)
            accidental_lure[k] = len(realised - intended_lure_sets[k])
            missing_lure[k] = len(intended_lure_sets[k] - realised)

        ok = (
            n_missing_match == 0
            and all(v == 0 for v in missing_lure.values())
            and (not self.avoid_accidental_matches or n_accidental_match == 0)
            and (
                not self.avoid_accidental_lures
                or all(v == 0 for v in accidental_lure.values())
            )
        )

        return {
            "n_match_intended": len(intended_match_set),
            "n_match_realised": len(realised_match_set),
            "n_accidental_match": n_accidental_match,
            "n_missing_match": n_missing_match,
            "n_lure_intended": {k: len(intended_lure_sets[k]) for k in self.lure_offsets},
            "n_lure_realised": n_lure_realised,
            "n_accidental_lure": accidental_lure,
            "n_missing_lure": missing_lure,
            "match_rate": len(realised_match_set) / max(1, L - n),
            "ok": ok,
        }
