# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Token duration statistics."""

import numpy as np
from collections import defaultdict


def token_duration_stats(
    sequence: list[str],
    durations: list[float] | None = None,
    frame_rate: float | None = None,
    summary_stats: bool = False,
) -> dict:
    """
    Compute duration statistics for tokens.

    Parameters
    ----------
    sequence : list of str
        List of tokens.
    durations : list of float, optional
        Duration for each token occurrence (must match sequence length).
    frame_rate : float, optional
        Frame rate (Hz) for frame-based sequences. If provided and durations
        is None, durations are computed from consecutive frame counts.
    summary_stats : bool, default=False
        If False, return dict {token: [d1, d2, ...]}
        If True, return dict {token: (mean, variance)}

    Returns
    -------
    dict
        Token duration statistics.

    Raises
    ------
    ValueError
        If neither durations nor frame_rate provided.
        If durations length doesn't match sequence length.

    Examples
    --------
    >>> seq = ['A', 'A', 'B', 'A']
    >>> durs = [0.5, 0.6, 0.3, 0.7]
    >>> token_duration_stats(seq, durs, summary_stats=True)
    {'A': (0.6, 0.01), 'B': (0.3, 0.0)}

    >>> seq_frames = ['A', 'A', 'A', 'B', 'B', 'A']
    >>> stats = token_duration_stats(seq_frames, frame_rate=30.0, summary_stats=True)
    """
    if durations is None and frame_rate is None:
        raise ValueError("Must provide either durations or frame_rate")

    if durations is not None:
        if len(durations) != len(sequence):
            raise ValueError(f"durations length ({len(durations)}) must match " f"sequence length ({len(sequence)})")
        duration_map = defaultdict(list)
        for token, dur in zip(sequence, durations):
            duration_map[token].append(dur)
    else:
        duration_map = _compute_frame_durations(sequence, frame_rate)

    if not summary_stats:
        return dict(duration_map)
    else:
        summary = {}
        for token, durs in duration_map.items():
            durs_arr = np.array(durs)
            summary[token] = (np.mean(durs_arr), np.var(durs_arr))
        return summary


def _compute_frame_durations(sequence: list[str], frame_rate: float) -> defaultdict:
    """
    Compute durations from frame-based sequence via run-length encoding.

    Parameters
    ----------
    sequence : list of str
        Frame-level token sequence.
    frame_rate : float
        Frames per second.

    Returns
    -------
    defaultdict
        Token -> list of durations (in seconds).
    """
    duration_map = defaultdict(list)

    if len(sequence) == 0:
        return duration_map

    current_token = sequence[0]
    run_length = 1

    for i in range(1, len(sequence)):
        if sequence[i] == current_token:
            run_length += 1
        else:
            duration_sec = run_length / frame_rate
            duration_map[current_token].append(duration_sec)
            current_token = sequence[i]
            run_length = 1

    duration_sec = run_length / frame_rate
    duration_map[current_token].append(duration_sec)

    return duration_map
