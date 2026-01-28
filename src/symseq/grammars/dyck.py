# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""
dyck.py

This module contains the DyckGenerator class for creating k-Dyck languages.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Literal
import numpy as np
import string


Mode = Literal["stack", "uniform"]


class DyckGenerator:
    """
    Dyck-language generator with variable-length (stack) and fixed-length uniform modes.

    Parameters
    ----------
    k : int
        Number of parenthesis types (creates a k-Dyck language).
        Default bracket pairs will be (a, A), (b, B), (c, C), etc.
    mode : {"stack", "uniform"}
        Generation mode.
        - "stack": variable-length strings from a stack process controlled by `p_open`.
        - "uniform": fixed-length, uniform-at-length strings via Catalan recursion; uses `target_pairs`.
    parentheses : dict[str, str], optional
        Custom mapping of opening → closing symbols. If provided, overrides the default
        brackets generated from `k`. For example: ``{"(" : ")", "[" : "]"}``.
    p_open : float, optional
        Probability of opening a new parenthesis while the stack is non-empty (stack mode only).
        Must satisfy ``0 < p_open < 0.5``. Ignored in uniform mode. Default ``0.3``.
    target_pairs : int, optional
        Number of pairs (i.e., length = ``2 * target_pairs``) for uniform mode.
        Must be a positive integer in uniform mode. Ignored in stack mode.
    max_depth : int, optional
        Maximum nesting depth allowed. If specified, generated strings will never have more
        than `max_depth` open brackets at any point. Strings violating this constraint are
        rejected and regenerated. Default ``None`` (no constraint).
    distractors : sequence of str, optional
        Custom inventory of string to insert as distractors. If not provided, defaults
        to digit strings '0', '1', ..., up to `n_distractors` symbols.
        Set to empty list ``[]`` to disable distractors. Default ``None`` (uses defaults).
    n_distractors : int, optional
        Number of default distractor symbols to generate (using digits 0-9).
        Only used when `distractors` is ``None``. Maximum 10. Default ``2``.
    rng : numpy.random.Generator, optional
        Random number generator. If ``None``, a default generator is created.
    uniform_colorize : bool, optional
        If ``True``, in uniform mode each pair's parenthesis *type* is chosen independently
        and uniformly from the available opening symbols. If you want a single-type Dyck language,
        pass ``k=1`` or set this to ``False``. Default ``True``.

    Attributes
    ----------
    alphabet : list of str
        Sorted list of all symbols this generator may emit: all opens, all closes,
        and (if present) the distractors.

    Notes
    -----
    - **k-Dyck language**: A Dyck language with `k` types of parentheses. For example,
      k=2 gives you two types like (a, A) and (b, B).
    - **Default distractors**: By default, the generator uses digit strings ('0', '1', ...)
      as distractors. These won't conflict with the default letter-based parentheses.
    - **Uniform-at-length** here means: the *Dyck structure* with `n = target_pairs` pairs is sampled
      uniformly among the `C_n` Catalan structures. If you have multiple kinds of parentheses and
      `uniform_colorize=True`, the *types* for each matched pair are chosen independently and uniformly,
      which yields uniform structure but not uniform over all colorings.
    - Outputs are lists of string.

    Examples
    --------
    >>> # Simple 2-Dyck language with stack mode
    >>> gen = DyckGenerator(k=2, mode="stack")
    >>> gen.generate_string()
    ['a', 'b', 'B', 'A']

    >>> # With default distractors
    >>> gen = DyckGenerator(k=1, mode="stack")
    >>> gen.generate_string(add_distractors=True)
    ['a', '0', 'a', 'A', '1', 'A']

    >>> # 1-Dyck with uniform mode and max depth
    >>> gen = DyckGenerator(k=1, mode="uniform", target_pairs=5, max_depth=2)
    >>> gen.generate_string()
    ['a', 'a', 'A', 'A', 'a', 'A', 'a', 'A', 'a', 'A']

    >>> # Custom brackets with no distractors
    >>> gen = DyckGenerator(k=2, mode="stack", parentheses={"(": ")", "[": "]"}, distractors=[])
    >>> gen.generate_string()
    ['(', '[', ']', ')']
    """

    def __init__(
        self,
        k: int,
        mode: Mode,
        parentheses: Optional[Dict[str, str]] = None,
        p_open: float = 0.3,
        target_pairs: Optional[int] = None,
        max_depth: Optional[int] = None,
        distractors: Optional[Sequence[str]] = None,
        n_distractors: int = 2,
        rng: Optional[np.random.Generator] = None,
        uniform_colorize: bool = True,
    ):
        self.k = k
        self.mode = mode
        self.p_open = p_open
        self.target_pairs = target_pairs
        self.max_depth = max_depth
        self.n_distractors = n_distractors
        self.rng = rng
        self.uniform_colorize = uniform_colorize

        # Validation
        if not isinstance(k, int) or k < 1:
            raise ValueError("`k` must be a positive integer.")

        # Generate or use provided parentheses
        if parentheses is not None:
            self.parentheses = parentheses
            if len(self.parentheses) != k:
                raise ValueError(f"Provided `parentheses` has {len(parentheses)} types but k={k}.")
        else:
            self.parentheses = self._generate_default_parentheses(k)

        if not self.parentheses:
            raise ValueError("`parentheses` must be a non-empty mapping of open -> close.")

        # Derived attributes
        self._opens = list(self.parentheses.keys())
        self._closes = list(self.parentheses.values())

        # Generate or use provided distractors
        if distractors is not None:
            self.distractors = list(distractors)
        else:
            self.distractors = self._generate_default_distractors(n_distractors)

        # Validate no collision between distractors and parentheses
        if self.distractors:
            paren_symbols = set(self._opens) | set(self._closes)
            distractor_symbols = set(self.distractors)
            collision = paren_symbols & distractor_symbols
            if collision:
                raise ValueError(
                    f"Distractors cannot overlap with parenthesis symbols. "
                    f"Collision found: {collision}. "
                    f"Provide custom distractors that don't conflict with your parentheses."
                )

        if self.rng is None:
            self.rng = np.random.default_rng()

        if self.mode == "stack":
            if not (0.0 < self.p_open < 0.5):
                raise ValueError("In 'stack' mode, p_open must be in (0, 0.5).")
        elif self.mode == "uniform":
            if not (isinstance(self.target_pairs, int) and self.target_pairs > 0):
                raise ValueError("In 'uniform' mode, target_pairs must be a positive integer.")
        else:
            raise ValueError("mode must be 'stack' or 'uniform'.")

        if self.max_depth is not None:
            if not (isinstance(self.max_depth, int) and self.max_depth > 0):
                raise ValueError("`max_depth` must be a positive integer or None.")

        self._compute_alphabet()

    def _generate_default_parentheses(self, k: int) -> Dict[str, str]:
        """
        Generate default parenthesis pairs using lowercase/uppercase letters.

        For k types, uses: (a, A), (b, B), (c, C), ...

        Parameters
        ----------
        k : int
            Number of parenthesis types.

        Returns
        -------
        dict[str, str]
            Mapping of opening to closing symbols.
        """
        if k > 26:
            raise ValueError("Default parentheses support up to k=26. Provide custom `parentheses` for larger k.")

        lowercase = string.ascii_lowercase[:k]
        uppercase = string.ascii_uppercase[:k]
        return {lower: upper for lower, upper in zip(lowercase, uppercase)}

    def _generate_default_distractors(self, n_distractors: int) -> List[str]:
        """
        Generate default distractor symbols using digit strings.

        For n distractors, uses: '0', '1', '2', ..., up to '9'.

        Parameters
        ----------
        n_distractors : int
            Number of default distractors to generate.

        Returns
        -------
        list of str
            List of distractor symbols.
        """
        if n_distractors <= 0:
            return []

        if n_distractors > 10:
            raise ValueError(
                f"Default distractors support up to 10 symbols (digits 0-9). "
                f"You requested {n_distractors}. "
                f"Provide custom `distractors` for more than 10 symbols."
            )

        return [str(i) for i in range(n_distractors)]

    def _check_max_depth(self, string: List[str]) -> bool:
        """
        Check if a token list respects the max_depth constraint.

        Parameters
        ----------
        string : list of str
            Token list to check.

        Returns
        -------
        bool
            True if max_depth is respected (or None), False otherwise.
        """
        if self.max_depth is None:
            return True

        depth = 0
        max_seen = 0
        for tok in string:
            if tok in self._opens:
                depth += 1
                max_seen = max(max_seen, depth)
                if max_seen > self.max_depth:
                    return False
            elif tok in self._closes:
                depth -= 1

        return True

    def _is_valid_dyck(self, string: List[str]) -> bool:
        """
        Check if a token list is a valid Dyck string (ignoring distractors).

        A valid Dyck string must:
        1. Have matching open/close pairs
        2. Never have a close before its matching open
        3. End with depth 0 (all brackets closed)

        Parameters
        ----------
        string : list of str
            Token list to check.

        Returns
        -------
        bool
            True if the token list forms a valid Dyck string, False otherwise.
        """
        stack: List[str] = []

        for tok in string:
            # Skip distractors
            if self.distractors and tok in self.distractors:
                continue

            if tok in self._opens:
                stack.append(tok)
            elif tok in self._closes:
                # Find which open this close corresponds to
                matching_open = None
                for open_sym, close_sym in self.parentheses.items():
                    if close_sym == tok:
                        matching_open = open_sym
                        break

                # Check if we can close (stack not empty and matches)
                if not stack or stack[-1] != matching_open:
                    return False
                stack.pop()
            # If token is neither open, close, nor distractor, it's invalid
            elif tok not in self._opens and tok not in self._closes:
                return False

        # All brackets must be closed
        return len(stack) == 0

    # ============================== Public API ==============================

    def generate_string(
        self,
        add_distractors: bool = False,
        n_distractors: int = 0,
        max_attempts: int = 1000,
    ) -> List[str]:
        """
        Generate a single Dyck token list.

        Parameters
        ----------
        add_distractors : bool, optional
            If ``True`` and distractors are available, insert distractors at random positions
            after generating a valid base string.
        n_distractors : int, optional
            Number of distractors to insert if ``add_distractors`` is ``True``.
            If ``0``, uses the default ``n_distractors`` set during initialization. Default ``0``.
        max_attempts : int, optional
            Maximum number of attempts to generate a string satisfying max_depth constraint.
            Default ``1000``.

        Returns
        -------
        list of str
            A valid Dyck token list; variable length in stack mode, fixed length (``2 * target_pairs``)
            in uniform mode.

        Raises
        ------
        RuntimeError
            If max_attempts is exceeded without finding a valid string.
        """
        # Use default n_distractors if not specified
        if add_distractors and n_distractors == 0:
            n_distractors = self.n_distractors

        for attempt in range(max_attempts):
            if self.mode == "stack":
                string = self._sample_valid_stack()
            else:
                string = self._sample_uniform_pairs(self.target_pairs or 1)

            if self._check_max_depth(string):
                if add_distractors:
                    string = self._inject_distractors(string, n_distractors)
                return string

        raise RuntimeError(
            f"Failed to generate a valid string satisfying max_depth={self.max_depth} "
            f"after {max_attempts} attempts. Consider increasing max_depth or max_attempts."
        )

    def generate_string_set(
        self,
        n: int,
        add_distractors: bool = False,
        n_distractors: int = 0,
        max_attempts: int = 1000,
    ) -> List[List[str]]:
        """
        Generate a list of Dyck token lists.

        Parameters
        ----------
        n : int
            Number of strings to generate.
        add_distractors : bool, optional
            If ``True``, add distractors into each string (requires a non-empty distractor inventory).
        n_distractors : int, optional
            Number of distractors to insert per string if ``add_distractors`` is ``True``.
        max_attempts : int, optional
            Maximum attempts per string to satisfy max_depth constraint.

        Returns
        -------
        list of list of str
            A list of token lists.
        """
        return [
            self.generate_string(
                add_distractors=add_distractors, n_distractors=n_distractors, max_attempts=max_attempts
            )
            for _ in range(n)
        ]

    def generate_nongrammatical_strings(
        self,
        n: int,
        n_deviants: int = 1,
        strategies: Optional[Sequence[str]] = None,
        add_distractors: bool = False,
        n_distractors: int = 0,
        max_attempts: int = 1000,
        verify_illegal: bool = True,
    ) -> List[List[str]]:
        """
        Generate illegal strings by applying local corruptions to valid Dyck strings.

        Parameters
        ----------
        n : int
            Number of illegal strings to generate.
        n_deviants : int, optional
            Number of corruption operations per string. Default ``1``.
        strategies : sequence of {{'replace_close','insert_extra_close','delete_token','swap_adjacent','truncate'}}, optional
            Allowed corruption strategies. If ``None``, a default mixture is used.
        add_distractors : bool, optional
            If ``True``, inject distractors after corruption.
        n_distractors : int, optional
            Number of distractors per string when ``add_distractors`` is ``True``.
            If ``0``, uses the default ``n_distractors`` set during initialization.
        max_attempts : int, optional
            Maximum attempts per base string to satisfy max_depth constraint and illegality.
        verify_illegal : bool, optional
            If ``True``, verify that the corrupted string is actually illegal and retry if not.
            Default ``True``.

        Returns
        -------
        list of list of str
            A list of token lists that violate Dyck well-formedness.

        Raises
        ------
        RuntimeError
            If max_attempts is exceeded without generating an illegal string.
        """
        if strategies is None:
            strategies = ("replace_close", "insert_extra_close", "delete_token", "swap_adjacent", "truncate")

        # Use default n_distractors if not specified
        if add_distractors and n_distractors == 0:
            n_distractors = self.n_distractors

        out: List[List[str]] = []
        for _ in range(n):
            for attempt in range(max_attempts):
                base = self.generate_string(add_distractors=False, max_attempts=max_attempts)
                string = base[:]

                # Apply violations
                for _k in range(max(0, n_deviants)):
                    strat = self._choice(np.array(strategies, dtype=object))
                    string = self._apply_violation(string, strat)

                # Check if result is actually illegal (if verification enabled)
                if not verify_illegal or not self._is_valid_dyck(string):
                    if add_distractors:
                        string = self._inject_distractors(string, n_distractors)
                    out.append(string)
                    break
            else:
                raise RuntimeError(
                    f"Failed to generate an illegal string after {max_attempts} attempts. "
                    f"Try increasing n_deviants or max_attempts."
                )

        return out

    # =========================== Core: stack mode ===========================

    def _sample_valid_stack(self) -> List[str]:
        """
        Sample a valid Dyck token list using a probabilistic stack process.
        """
        string: List[str] = []
        stack: List[str] = []

        # start with one random opening
        first_open = self._choice(self._opens)
        string.append(first_open)
        stack.append(first_open)

        while stack:
            if self.rng.random() < self.p_open:
                op = self._choice(self._opens)
                string.append(op)
                stack.append(op)
            else:
                op = stack.pop()
                cl = self.parentheses[op]
                string.append(cl)

        return string

    # ========================= Core: uniform mode ==========================

    def _sample_uniform_pairs(self, n_pairs: int) -> List[str]:
        """
        Sample a valid Dyck token list with exactly `2 * n_pairs` string,
        uniformly over Catalan structures of size `n_pairs`. If multiple
        parenthesis types are available and `uniform_colorize=True`, each pair's
        type is chosen independently and uniformly.
        """
        if n_pairs == 0:
            return []

        # Precompute Catalan numbers up to n_pairs
        C = self._catalans(n_pairs)

        def recurse(n: int) -> List[str]:
            if n == 0:
                return []
            # choose split k with probability (C[k] * C[n-1-k]) / C[n]
            weights = np.fromiter((C[k] * C[n - 1 - k] for k in range(n)), dtype=np.int64)
            k = int(self._weighted_choice(weights))
            # choose parenthesis type for the outermost pair
            op = self._choice(self._opens) if self.uniform_colorize else self._opens[0]
            cl = self.parentheses[op]
            left = recurse(k)
            right = recurse(n - 1 - k)
            return [op] + left + [cl] + right

        return recurse(n_pairs)

    # ===================== Violations & Distractors ========================

    def _apply_violation(self, string: List[str], strategy: str) -> List[str]:
        """
        Apply a single corruption strategy to a token list.
        """
        if not string:
            return string[:]

        t = string[:]

        if strategy == "replace_close":
            close_positions = [i for i, tok in enumerate(t) if tok in self._closes]
            if close_positions:
                idx = int(self._choice(close_positions))
                wrong = [c for c in self._closes if c != t[idx]]
                if wrong:
                    t[idx] = self._choice(wrong)
            return t

        if strategy == "insert_extra_close":
            pos = int(self.rng.integers(0, len(t) + 1))
            t.insert(pos, self._choice(self._closes))
            return t

        if strategy == "delete_token":
            pos = int(self.rng.integers(0, len(t)))
            del t[pos]
            return t

        if strategy == "swap_adjacent":
            if len(t) >= 2:
                pos = int(self.rng.integers(0, len(t) - 1))
                t[pos], t[pos + 1] = t[pos + 1], t[pos]
            return t

        if strategy == "truncate":
            if len(t) > 1:
                cut = int(self.rng.integers(1, len(t)))
                t = t[:cut]
            return t

        return t  # unknown strategy: no-op

    def _inject_distractors(self, string: List[str], n_distractors: int) -> List[str]:
        """
        Insert `n_distractors` distractor string at random positions.
        """
        if not self.distractors or n_distractors <= 0:
            return string[:]
        t = string[:]
        for _ in range(n_distractors):
            pos = int(self.rng.integers(0, len(t) + 1))
            sym = self._choice(self.distractors)
            t.insert(pos, sym)
        return t

    # ============================== Utilities ==============================

    def _compute_alphabet(self) -> None:
        """
        Compute the generator's alphabet from parentheses and distractors.
        """
        items = set(self._opens) | set(self._closes)
        if self.distractors:
            items |= set(self.distractors)
        self.alphabet = sorted(items)

    def _catalans(self, n: int) -> List[int]:
        """
        Compute Catalan numbers C[0..n] via dynamic programming.

        Returns
        -------
        list of int
            Catalan numbers from 0 to n inclusive.
        """
        C = [0] * (n + 1)
        C[0] = 1
        for m in range(1, n + 1):
            s = 0
            # C_m = sum_{k=0}^{m-1} C_k * C_{m-1-k}
            for k in range(m):
                s += C[k] * C[m - 1 - k]
            C[m] = s
        return C

    def _weighted_choice(self, weights: np.ndarray) -> int:
        """
        Draw an index according to nonnegative integer weights.
        """
        total = weights.sum()
        # numerical guard: if all zero (shouldn't happen), fall back to uniform
        if total <= 0:
            return int(self.rng.integers(0, len(weights)))
        r = self.rng.integers(0, total)
        acc = 0
        for i, w in enumerate(weights):
            acc += int(w)
            if r < acc:
                return i
        return len(weights) - 1  # fallback

    def _choice(self, seq: Sequence) -> any:
        """
        RNG-friendly choice over an arbitrary Python sequence.
        """
        return seq[int(self.rng.integers(0, len(seq)))]

    def __repr__(self) -> str:
        """String representation of the generator."""
        return (
            f"DyckGenerator(k={self.k}, mode={self.mode!r}, "
            f"parentheses={self.parentheses!r}, "
            f"p_open={self.p_open}, "
            f"target_pairs={self.target_pairs}, "
            f"max_depth={self.max_depth}, "
            f"distractors={self.distractors!r}, "
            f"n_distractors={self.n_distractors}, "
            f"uniform_colorize={self.uniform_colorize})"
        )
