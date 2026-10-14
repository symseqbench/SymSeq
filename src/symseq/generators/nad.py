# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Generators for simple, crossed, and nested non-adjacent dependencies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import factorial
from typing import ClassVar

import numpy as np

from symseq.core.sequencer import SymbolicSequencer
from symseq.generators.registry import register
from symseq.trial import Target, Trial
from symseq.utils.io import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class _Frame:
    """Internal representation of one generated dependency frame."""

    symbols: list[str]
    pair_index: int | None = None
    dependency_order: tuple[int, ...] | None = None
    dependency_length: int | None = None


def _nonnegative_int(value: object, name: str) -> int:
    """Validate and return a non-negative integer."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a non-negative integer.")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return parsed


def _positive_int(value: object, name: str) -> int:
    """Validate and return a positive integer."""
    parsed = _nonnegative_int(value, name)
    if parsed == 0:
        raise ValueError(f"{name} must be a positive integer.")
    return parsed


def _fraction(value: object, name: str) -> float:
    """Validate and return a finite fraction in the closed unit interval."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a finite number in [0, 1].")
    parsed = float(value)
    if not np.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1].")
    return parsed


def _boolean(value: object, name: str) -> bool:
    """Validate and return a Boolean value."""
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a bool.")
    return bool(value)


def _symbol(value: object, name: str) -> str:
    """Validate and return a non-empty plain Python string."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return str(value)


def _resolve_dependency_pairs(
    n_deps: int | None,
    dependency_pairs: Sequence[tuple[str, str]] | None,
) -> tuple[tuple[str, str], ...]:
    """Resolve generated or explicit dependency pairs and validate their roles."""
    if dependency_pairs is None:
        resolved_n_deps = 1 if n_deps is None else _positive_int(n_deps, "n_deps")
        pairs = tuple((f"A{index}", f"B{index}") for index in range(resolved_n_deps))
    else:
        raw_pairs = list(dependency_pairs)
        if not raw_pairs:
            raise ValueError("dependency_pairs must contain at least one pair.")

        parsed_pairs: list[tuple[str, str]] = []
        for index, pair in enumerate(raw_pairs):
            if isinstance(pair, (str, bytes)):
                raise ValueError(f"dependency_pairs[{index}] must contain exactly two symbols.")
            try:
                pair_items = list(pair)
            except TypeError as exc:
                raise ValueError(f"dependency_pairs[{index}] must contain exactly two symbols.") from exc
            if len(pair_items) != 2:
                raise ValueError(f"dependency_pairs[{index}] must contain exactly two symbols.")
            start = _symbol(pair_items[0], f"dependency_pairs[{index}][0]")
            terminal = _symbol(pair_items[1], f"dependency_pairs[{index}][1]")
            parsed_pairs.append((start, terminal))
        pairs = tuple(parsed_pairs)

        if n_deps is not None and _positive_int(n_deps, "n_deps") != len(pairs):
            raise ValueError("n_deps must match the number of dependency_pairs.")

    if len(set(pairs)) != len(pairs):
        raise ValueError("dependency_pairs must be unique.")

    starts = [start for start, _ in pairs]
    terminals = [terminal for _, terminal in pairs]
    if len(set(starts)) != len(starts):
        raise ValueError("Dependency start symbols must be unique.")
    if len(set(terminals)) != len(terminals):
        raise ValueError("Dependency terminal symbols must be unique.")
    if set(starts) & set(terminals):
        raise ValueError("Dependency start and terminal symbols must be disjoint.")
    return pairs


class _NonAdjacentDependenciesBase(SymbolicSequencer, ABC):
    """Shared validation, finite sampling, violation, and Trial infrastructure."""

    intrinsic_target_granularities: ClassVar[dict[str, str]] = {"grammaticality": "per_trial"}

    def __init__(
        self,
        *,
        label: str,
        n_deps: int | None,
        dependency_pairs: Sequence[tuple[str, str]] | None,
        extra_symbols: Sequence[str] = (),
        eos: str = "#",
        rng: np.random.Generator | None = None,
        seed: int | None = None,
        verbose: bool = True,
    ) -> None:
        """
        Initialize shared non-adjacent dependency state.

        Parameters
        ----------
        label : str
            Generator label.
        n_deps : int or None
            Number of generated dependency pairs when ``dependency_pairs`` is omitted.
        dependency_pairs : sequence of tuple of str or None
            Explicit start-terminal pairs.
        extra_symbols : sequence of str, optional
            Additional symbol roles, such as fillers.
        eos : str, optional
            End-of-sequence symbol. It is not emitted in generated frames.
        rng : numpy.random.Generator or None, optional
            Random number generator. Takes precedence over ``seed``.
        seed : int or None, optional
            Seed used when ``rng`` is omitted.
        verbose : bool, optional
            Whether to log construction details.
        """
        self.label = _symbol(label, "label")
        self.dependency_pairs = _resolve_dependency_pairs(n_deps, dependency_pairs)
        self.n_deps = len(self.dependency_pairs)
        self.eos = _symbol(eos, "eos")

        parsed_extra_symbols = tuple(_symbol(item, "extra symbol") for item in extra_symbols)
        dependency_symbols = {symbol for pair in self.dependency_pairs for symbol in pair}
        if dependency_symbols & set(parsed_extra_symbols):
            raise ValueError("Dependency symbols and extra symbols must be disjoint.")
        if self.eos in dependency_symbols or self.eos in parsed_extra_symbols:
            raise ValueError("eos must not overlap with a generated symbol.")

        if rng is not None and not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator or None.")
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        if verbose and rng is None and seed is None:
            logger.warning("%s sequences will not be reproducible.", type(self).__name__)

        alphabet = sorted(dependency_symbols | set(parsed_extra_symbols))
        super().__init__(
            label=self.label,
            alphabet_size=len(alphabet),
            alphabet=alphabet,
            rng=self.rng,
            verbose=verbose,
        )

        self.start_symbols = [start for start, _ in self.dependency_pairs]
        self.terminal_symbols = [terminal for _, terminal in self.dependency_pairs]

        if verbose:
            self.print()

    def print(self) -> None:
        """Log the generator's resolved symbol roles."""
        logger.info("***************************************************************************")
        logger.info("Non-adjacent dependency generator: %s", self.label)
        logger.info("Paradigm: %s", type(self).__name__)
        logger.info("Alphabet: %s", self.alphabet)
        logger.info("Dependency pairs: %s", self.dependency_pairs)

    @abstractmethod
    def _apply_violation(self, frame: _Frame) -> None:
        """Modify a generated frame in place so it violates the subclass rule."""

    def _sample_one(
        self,
        support_size: int,
        frame_from_rank: Callable[[int], _Frame],
        *,
        violation: bool,
    ) -> tuple[_Frame, bool]:
        """Sample one frame uniformly from a finite support."""
        parsed_violation = _boolean(violation, "violation")
        rank = int(self.rng.integers(support_size))
        frame = frame_from_rank(rank)
        if parsed_violation:
            self._require_violation_support()
            self._apply_violation(frame)
        return frame, not parsed_violation

    def _sample_batch(
        self,
        n_samples: int,
        support_size: int,
        frame_from_rank: Callable[[int], _Frame],
        *,
        frac_violations: float,
        replace: bool,
        strict: bool,
    ) -> tuple[list[_Frame], list[bool]]:
        """Sample a batch and apply an exact floor-rounded violation fraction."""
        requested = _nonnegative_int(n_samples, "n_samples")
        fraction = _fraction(frac_violations, "frac_violations")
        parsed_replace = _boolean(replace, "replace")
        parsed_strict = _boolean(strict, "strict")

        actual = requested
        if not parsed_replace and requested > support_size:
            if parsed_strict:
                raise ValueError(f"Cannot generate {requested} unique strings from a support of size {support_size}.")
            actual = support_size
            logger.warning(
                "Requested %d unique strings from a support of size %d; returning the complete support.",
                requested,
                support_size,
            )

        if support_size > np.iinfo(np.int64).max:
            raise ValueError("The configured finite support is too large to sample by rank.")

        if actual == 0:
            ranks: list[int] = []
        else:
            sampled = self.rng.choice(support_size, size=actual, replace=parsed_replace)
            ranks = [int(rank) for rank in np.atleast_1d(sampled)]
        frames = [frame_from_rank(rank) for rank in ranks]

        grammaticality = [True] * actual
        n_violations = int(np.floor(fraction * actual))
        if n_violations:
            self._require_violation_support()
            violation_indices = self.rng.choice(actual, size=n_violations, replace=False)
            for raw_index in np.atleast_1d(violation_indices):
                index = int(raw_index)
                self._apply_violation(frames[index])
                grammaticality[index] = False

        return frames, grammaticality

    def _require_violation_support(self) -> None:
        """Raise when the configured pair set cannot express a violation."""
        if self.n_deps < 2:
            raise ValueError("At least two dependency pairs are required to generate a violation.")

    def _intrinsic_targets(self, frame: _Frame, grammatical: bool) -> dict[str, Target]:
        """Build intrinsic targets shared by every NAD variant."""
        return {"grammaticality": Target(values=grammatical, mask=None, granularity="per_trial")}

    def _to_trial(self, frame: _Frame, grammatical: bool) -> Trial:
        """Convert an internal frame to the public Trial representation."""
        meta: dict[str, object] = {
            "paradigm": type(self).__name__,
            "label": self.label,
            "n_deps": self.n_deps,
            "length": len(frame.symbols),
            "grammatical": grammatical,
        }
        if frame.dependency_length is not None:
            meta["dependency_length"] = frame.dependency_length
        if frame.dependency_order is not None:
            meta["dependency_order"] = list(frame.dependency_order)

        return Trial(
            symbols=list(frame.symbols),
            meta=meta,
            intrinsic_targets=self._intrinsic_targets(frame, grammatical),
        )


@register("NonAdjacentDependencies")
class NonAdjacentDependencies(_NonAdjacentDependenciesBase):
    """Generate frames of the form ``A_i X...X B_i``."""

    intrinsic_target_granularities: ClassVar[dict[str, str]] = {
        **_NonAdjacentDependenciesBase.intrinsic_target_granularities,
        "pair_index": "per_trial",
    }

    def __init__(
        self,
        label: str = "Default_NAD",
        n_deps: int | None = None,
        n_unique_fillers: int = 1,
        dependency_pairs: Sequence[tuple[str, str]] | None = None,
        fillers: Sequence[str] | None = None,
        eos: str = "#",
        rng: np.random.Generator | None = None,
        seed: int | None = None,
        verbose: bool = True,
    ) -> None:
        """
        Initialize a simple non-adjacent dependency generator.

        Parameters
        ----------
        label : str, optional
            Generator label.
        n_deps : int or None, optional
            Number of generated dependency pairs. Defaults to one when pairs are omitted.
        n_unique_fillers : int, optional
            Number of generated filler symbols when ``fillers`` is omitted.
        dependency_pairs : sequence of tuple of str or None, optional
            Explicit start-terminal dependency pairs.
        fillers : sequence of str or None, optional
            Explicit filler symbols. This takes precedence over ``n_unique_fillers``.
        eos : str, optional
            End-of-sequence symbol. It is not emitted in generated frames.
        rng : numpy.random.Generator or None, optional
            Random number generator. Takes precedence over ``seed``.
        seed : int or None, optional
            Seed used when ``rng`` is omitted.
        verbose : bool, optional
            Whether to log construction details.
        """
        if fillers is None:
            filler_count = _nonnegative_int(n_unique_fillers, "n_unique_fillers")
            parsed_fillers = tuple(f"X{index}" for index in range(filler_count))
        else:
            parsed_fillers = tuple(_symbol(filler, "filler") for filler in fillers)
            if len(set(parsed_fillers)) != len(parsed_fillers):
                raise ValueError("fillers must be unique.")

        self.fillers = parsed_fillers
        self.n_unique_fillers: int = len(parsed_fillers)
        super().__init__(
            label=label,
            n_deps=n_deps,
            dependency_pairs=dependency_pairs,
            extra_symbols=self.fillers,
            eos=eos,
            rng=rng,
            seed=seed,
            verbose=verbose,
        )

    def _generation_parameters(self, filler_len: int, randomize_fillers: bool) -> tuple[int, bool]:
        """Validate per-generation filler parameters."""
        return _nonnegative_int(filler_len, "filler_len"), _boolean(randomize_fillers, "randomize_fillers")

    def _filler_pattern_count(self, filler_len: int, randomize_fillers: bool) -> int:
        """Return the number of distinct filler patterns for one dependency pair."""
        if filler_len == 0 or not self.fillers:
            return 1
        if randomize_fillers:
            return int(self.n_unique_fillers**filler_len)
        return self.n_unique_fillers

    def _filler_pattern(self, rank: int, filler_len: int, randomize_fillers: bool) -> list[str]:
        """Decode one filler-pattern rank into concrete symbols."""
        if filler_len == 0 or not self.fillers:
            return []
        if not randomize_fillers:
            return [self.fillers[rank]] * filler_len

        fillers = [""] * filler_len
        for position in range(filler_len - 1, -1, -1):
            rank, filler_index = divmod(rank, self.n_unique_fillers)
            fillers[position] = self.fillers[filler_index]
        return fillers

    def _support(self, filler_len: int, randomize_fillers: bool) -> tuple[int, Callable[[int], _Frame]]:
        """Return support cardinality and its rank decoder."""
        pattern_count = self._filler_pattern_count(filler_len, randomize_fillers)

        def frame_from_rank(rank: int) -> _Frame:
            pair_index, pattern_rank = divmod(rank, pattern_count)
            start, terminal = self.dependency_pairs[pair_index]
            filler_pattern = self._filler_pattern(pattern_rank, filler_len, randomize_fillers)
            return _Frame(
                symbols=[start, *filler_pattern, terminal],
                pair_index=pair_index,
                dependency_length=len(filler_pattern),
            )

        return self.n_deps * pattern_count, frame_from_rank

    def _apply_violation(self, frame: _Frame) -> None:
        """Replace a frame's expected terminal with a different terminal."""
        expected_terminal = frame.symbols[-1]
        candidates = [terminal for terminal in self.terminal_symbols if terminal != expected_terminal]
        frame.symbols[-1] = candidates[int(self.rng.integers(len(candidates)))]

    def _intrinsic_targets(self, frame: _Frame, grammatical: bool) -> dict[str, Target]:
        """Add the selected dependency pair index to common intrinsic targets."""
        targets = super()._intrinsic_targets(frame, grammatical)
        targets["pair_index"] = Target(values=frame.pair_index, mask=None, granularity="per_trial")
        return targets

    def generate_string(
        self,
        filler_len: int = 1,
        randomize_fillers: bool = False,
        violation: bool = False,
    ) -> list[str]:
        """
        Generate one simple dependency frame.

        Parameters
        ----------
        filler_len : int, optional
            Number of intervening filler symbols.
        randomize_fillers : bool, optional
            If ``False``, repeat one filler; otherwise sample independent filler positions.
        violation : bool, optional
            Whether to emit a mismatched terminal.

        Returns
        -------
        list of str
            Generated symbols.
        """
        filler_len, randomize_fillers = self._generation_parameters(filler_len, randomize_fillers)
        support_size, frame_from_rank = self._support(filler_len, randomize_fillers)
        frame, _ = self._sample_one(support_size, frame_from_rank, violation=violation)
        return frame.symbols

    def generate_vocabulary(
        self,
        filler_len: int = 1,
        randomize_fillers: bool = False,
        verbose: bool = True,
    ) -> list[list[str]]:
        """
        Enumerate every distinct grammatical frame for the requested filler policy.

        Parameters
        ----------
        filler_len : int, optional
            Number of intervening filler symbols.
        randomize_fillers : bool, optional
            Whether filler positions vary independently.
        verbose : bool, optional
            Whether to log enumeration.

        Returns
        -------
        list of list of str
            Complete grammatical support.
        """
        filler_len, randomize_fillers = self._generation_parameters(filler_len, randomize_fillers)
        if verbose:
            logger.info("Enumerating the complete support for %s.", self.label)
        support_size, frame_from_rank = self._support(filler_len, randomize_fillers)
        return [frame_from_rank(rank).symbols for rank in range(support_size)]

    def generate_string_set(
        self,
        n_samples: int,
        filler_len: int = 1,
        randomize_fillers: bool = False,
        frac_violations: float = 0.0,
        replace: bool = True,
        strict: bool = True,
    ) -> tuple[list[list[str]], list[bool]]:
        """
        Generate a labeled batch of simple dependency frames.

        Parameters
        ----------
        n_samples : int
            Requested number of frames.
        filler_len : int, optional
            Number of intervening filler symbols.
        randomize_fillers : bool, optional
            Whether filler positions vary independently.
        frac_violations : float, optional
            Fraction of the returned batch made ungrammatical, rounded down.
        replace : bool, optional
            Whether valid configurations may repeat.
        strict : bool, optional
            Whether oversubscribed without-replacement requests raise an error.

        Returns
        -------
        tuple of list of list of str and list of bool
            Generated frames and aligned grammaticality labels.
        """
        filler_len, randomize_fillers = self._generation_parameters(filler_len, randomize_fillers)
        support_size, frame_from_rank = self._support(filler_len, randomize_fillers)
        frames, grammaticality = self._sample_batch(
            n_samples,
            support_size,
            frame_from_rank,
            frac_violations=frac_violations,
            replace=replace,
            strict=strict,
        )
        return [frame.symbols for frame in frames], grammaticality

    def generate_trial(
        self,
        filler_len: int = 1,
        randomize_fillers: bool = False,
        violation: bool = False,
    ) -> Trial:
        """
        Generate one Trial with grammaticality and dependency-pair targets.

        Parameters
        ----------
        filler_len : int, optional
            Number of intervening filler symbols.
        randomize_fillers : bool, optional
            Whether filler positions vary independently.
        violation : bool, optional
            Whether to emit a mismatched terminal.

        Returns
        -------
        Trial
            Generated Trial.
        """
        filler_len, randomize_fillers = self._generation_parameters(filler_len, randomize_fillers)
        support_size, frame_from_rank = self._support(filler_len, randomize_fillers)
        frame, grammatical = self._sample_one(support_size, frame_from_rank, violation=violation)
        return self._to_trial(frame, grammatical)

    def generate_trials(
        self,
        n: int,
        filler_len: int = 1,
        randomize_fillers: bool = False,
        frac_violations: float = 0.0,
        replace: bool = True,
        strict: bool = True,
    ) -> list[Trial]:
        """
        Generate a Trial batch with an exact floor-rounded violation fraction.

        Parameters
        ----------
        n : int
            Requested number of Trials.
        filler_len : int, optional
            Number of intervening filler symbols.
        randomize_fillers : bool, optional
            Whether filler positions vary independently.
        frac_violations : float, optional
            Fraction of returned Trials made ungrammatical, rounded down.
        replace : bool, optional
            Whether valid configurations may repeat.
        strict : bool, optional
            Whether oversubscribed without-replacement requests raise an error.

        Returns
        -------
        list of Trial
            Generated Trials.
        """
        filler_len, randomize_fillers = self._generation_parameters(filler_len, randomize_fillers)
        support_size, frame_from_rank = self._support(filler_len, randomize_fillers)
        frames, grammaticality = self._sample_batch(
            n,
            support_size,
            frame_from_rank,
            frac_violations=frac_violations,
            replace=replace,
            strict=strict,
        )
        return [self._to_trial(frame, grammatical) for frame, grammatical in zip(frames, grammaticality, strict=True)]


class _OrderedNonAdjacentDependencies(_NonAdjacentDependenciesBase):
    """Shared implementation for crossed and nested dependency orderings."""

    _reverse_terminals: ClassVar[bool]
    _default_label: ClassVar[str]

    def __init__(
        self,
        label: str | None = None,
        n_deps: int | None = None,
        dependency_pairs: Sequence[tuple[str, str]] | None = None,
        eos: str = "#",
        rng: np.random.Generator | None = None,
        seed: int | None = None,
        verbose: bool = True,
    ) -> None:
        """
        Initialize an ordered non-adjacent dependency generator.

        Parameters
        ----------
        label : str or None, optional
            Generator label. Uses the concrete variant's default when omitted.
        n_deps : int or None, optional
            Number of generated dependency pairs. Defaults to one when pairs are omitted.
        dependency_pairs : sequence of tuple of str or None, optional
            Explicit start-terminal dependency pairs.
        eos : str, optional
            End-of-sequence symbol. It is not emitted in generated frames.
        rng : numpy.random.Generator or None, optional
            Random number generator. Takes precedence over ``seed``.
        seed : int or None, optional
            Seed used when ``rng`` is omitted.
        verbose : bool, optional
            Whether to log construction details.
        """
        super().__init__(
            label=self._default_label if label is None else label,
            n_deps=n_deps,
            dependency_pairs=dependency_pairs,
            eos=eos,
            rng=rng,
            seed=seed,
            verbose=verbose,
        )

    @staticmethod
    def _permutation_from_rank(rank: int, size: int) -> tuple[int, ...]:
        """Decode a lexicographic permutation rank using factoradics."""
        remaining = list(range(size))
        permutation: list[int] = []
        for width in range(size, 0, -1):
            block_size = factorial(width - 1)
            item_index, rank = divmod(rank, block_size)
            permutation.append(remaining.pop(item_index))
        return tuple(permutation)

    def _support(self) -> tuple[int, Callable[[int], _Frame]]:
        """Return permutation support cardinality and rank decoder."""

        def frame_from_rank(rank: int) -> _Frame:
            dependency_order = self._permutation_from_rank(rank, self.n_deps)
            terminal_order = dependency_order[::-1] if self._reverse_terminals else dependency_order
            starts = [self.start_symbols[index] for index in dependency_order]
            terminals = [self.terminal_symbols[index] for index in terminal_order]
            return _Frame(symbols=[*starts, *terminals], dependency_order=dependency_order)

        return factorial(self.n_deps), frame_from_rank

    def _apply_violation(self, frame: _Frame) -> None:
        """Swap two terminals to violate the required ordering."""
        first, second = [int(index) for index in self.rng.choice(self.n_deps, size=2, replace=False)]
        first += self.n_deps
        second += self.n_deps
        frame.symbols[first], frame.symbols[second] = frame.symbols[second], frame.symbols[first]

    def generate_string(self, violation: bool = False) -> list[str]:
        """
        Generate one ordered dependency frame.

        Parameters
        ----------
        violation : bool, optional
            Whether to swap two terminals and violate the ordering.

        Returns
        -------
        list of str
            Generated symbols.
        """
        support_size, frame_from_rank = self._support()
        frame, _ = self._sample_one(support_size, frame_from_rank, violation=violation)
        return frame.symbols

    def generate_string_set(
        self,
        n_samples: int,
        frac_violations: float = 0.0,
        replace: bool = True,
        strict: bool = True,
    ) -> tuple[list[list[str]], list[bool]]:
        """
        Generate a labeled batch of ordered dependency frames.

        Parameters
        ----------
        n_samples : int
            Requested number of frames.
        frac_violations : float, optional
            Fraction of returned frames made ungrammatical, rounded down.
        replace : bool, optional
            Whether dependency orders may repeat.
        strict : bool, optional
            Whether oversubscribed without-replacement requests raise an error.

        Returns
        -------
        tuple of list of list of str and list of bool
            Generated frames and aligned grammaticality labels.
        """
        support_size, frame_from_rank = self._support()
        frames, grammaticality = self._sample_batch(
            n_samples,
            support_size,
            frame_from_rank,
            frac_violations=frac_violations,
            replace=replace,
            strict=strict,
        )
        return [frame.symbols for frame in frames], grammaticality

    def generate_trial(self, violation: bool = False) -> Trial:
        """
        Generate one ordered-dependency Trial.

        Parameters
        ----------
        violation : bool, optional
            Whether to swap two terminals and violate the ordering.

        Returns
        -------
        Trial
            Generated Trial with grammaticality and dependency-order metadata.
        """
        support_size, frame_from_rank = self._support()
        frame, grammatical = self._sample_one(support_size, frame_from_rank, violation=violation)
        return self._to_trial(frame, grammatical)

    def generate_trials(
        self,
        n: int,
        frac_violations: float = 0.0,
        replace: bool = True,
        strict: bool = True,
    ) -> list[Trial]:
        """
        Generate ordered-dependency Trials with an exact violation fraction.

        Parameters
        ----------
        n : int
            Requested number of Trials.
        frac_violations : float, optional
            Fraction of returned Trials made ungrammatical, rounded down.
        replace : bool, optional
            Whether dependency orders may repeat.
        strict : bool, optional
            Whether oversubscribed without-replacement requests raise an error.

        Returns
        -------
        list of Trial
            Generated Trials.
        """
        support_size, frame_from_rank = self._support()
        frames, grammaticality = self._sample_batch(
            n,
            support_size,
            frame_from_rank,
            frac_violations=frac_violations,
            replace=replace,
            strict=strict,
        )
        return [self._to_trial(frame, grammatical) for frame, grammatical in zip(frames, grammaticality, strict=True)]


@register("CrossedNonAdjacentDependencies")
class CrossedNonAdjacentDependencies(_OrderedNonAdjacentDependencies):
    """Generate crossed frames ``A_i A_j ... B_i B_j ...``."""

    _reverse_terminals = False
    _default_label = "Default_Crossed_NAD"


@register("NestedNonAdjacentDependencies")
class NestedNonAdjacentDependencies(_OrderedNonAdjacentDependencies):
    """Generate nested frames ``A_i A_j ... B_j B_i ...``."""

    _reverse_terminals = True
    _default_label = "Default_Nested_NAD"
