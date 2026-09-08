# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""
cfg.py

Generic context-free grammar generator built on NLTK's PCFG definition scheme.

Unlike :class:`~symseq.generators.ag.artificial_grammar.ArtificialGrammar` (a
probabilistic finite-state automaton, limited to *regular* languages), this
generator accepts an arbitrary context-free grammar and samples strings from it
via stochastic top-down derivation. Grammaticality is checked with NLTK's
chart parser, so the same grammar serves as both the generator and the oracle.

Grammars are written in NLTK's standard format::

    S -> 'a' S 'b' [0.5] | 'a' 'b' [0.5]

Probabilities in ``[...]`` are optional: if omitted, productions sharing a
left-hand side are assigned uniform probabilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Sequence

import numpy as np

from symseq.core.sequencer import SymbolicSequencer
from symseq.generators.registry import register
from symseq.trial import Target, Trial

try:
    from nltk import CFG, PCFG, Nonterminal
    from nltk.grammar import ProbabilisticProduction
    from nltk.parse import ChartParser

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


@register("CFG")
class CFGGenerator(SymbolicSequencer):
    """
    Context-free grammar generator backed by an NLTK (P)CFG.

    Parameters
    ----------
    grammar : str or nltk.PCFG or nltk.CFG
        The grammar definition. May be:
        - an NLTK grammar-format string, e.g. ``"S -> 'a' S 'b' [0.5] | 'a' 'b' [0.5]"``;
        - a path to a ``.cfg``/``.pcfg``/``.txt`` file containing such a string;
        - an already-constructed ``nltk.PCFG`` or ``nltk.CFG`` object.
        Plain CFGs (productions without ``[prob]``) are converted to a PCFG with
        uniform probabilities per left-hand side.
    start_symbol : str, optional
        Reserved for API symmetry. NLTK infers the start symbol from the first
        production's left-hand side (conventionally ``S``); this argument is not
        used to override it. Default ``"S"``.
    label : str, optional
        Human-readable name for the grammar (carried into ``Trial.meta``).
        Default ``"CFG"``; presets set this to the preset name.
    max_length : int, optional
        Maximum number of terminals in a generated string. Derivations exceeding
        this are rejected and resampled. Default ``50``.
    max_depth : int, optional
        Maximum derivation depth (leftmost-expansion steps before forcing
        termination preference). Default ``100``.
    max_attempts : int, optional
        Maximum resampling attempts before raising. Default ``1000``.
    rng : numpy.random.Generator, optional
        Seeded RNG. If ``None``, one is created from ``seed``.
    seed : int, optional
        Seed used only when ``rng`` is not provided.

    Attributes
    ----------
    alphabet : list of str
        Sorted list of terminal symbols the grammar can emit.
    grammar : nltk.PCFG
        The parsed probabilistic grammar.

    Examples
    --------
    >>> gen = CFGGenerator("S -> 'a' S 'b' [0.5] | 'a' 'b' [0.5]", seed=0)
    >>> gen.generate_string()
    ['a', 'a', 'b', 'b']
    >>> gen.is_grammatical(['a', 'a', 'b', 'b'])
    True
    >>> gen.is_grammatical(['a', 'b', 'b'])
    False
    """

    intrinsic_target_granularities: ClassVar[dict[str, str]] = {"grammaticality": "per_trial"}

    def __init__(
        self,
        grammar: "str | PCFG | CFG",
        start_symbol: str = "S",
        label: str = "CFG",
        max_length: int = 50,
        max_depth: int = 100,
        max_attempts: int = 1000,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
    ):
        if not NLTK_AVAILABLE:
            raise ImportError("NLTK is required for the CFG generator. Install with: pip install nltk")

        self.max_length = max_length
        self.max_depth = max_depth
        self.max_attempts = max_attempts

        if rng is None:
            rng = np.random.default_rng(seed)

        self.grammar = self._build_grammar(grammar)
        self._start = self.grammar.start()
        # Cache a chart parser to use the grammar as its own grammaticality oracle.
        self._parser = ChartParser(self.grammar)

        # Index productions by LHS for fast weighted sampling during derivation.
        self._productions_by_lhs: dict = {}
        for prod in self.grammar.productions():
            self._productions_by_lhs.setdefault(prod.lhs(), []).append(prod)

        alphabet = self._extract_terminals(self.grammar)

        super().__init__(label=label, alphabet=alphabet, rng=rng, verbose=False)

    @classmethod
    def from_preset(cls, preset_name: str, seed: int | None = None, **overrides) -> "CFGGenerator":
        """
        Create a CFGGenerator from a named preset.

        Presets live in :mod:`symseq.generators.presets.cfg`; see
        :func:`symseq.generators.presets.cfg.list_presets` for the available names.

        Parameters
        ----------
        preset_name : str
            Name of the preset (e.g. ``"anbn"``, ``"dyck1"``, ``"palindrome"``).
        seed : int, optional
            Seed for the generator's RNG.
        **overrides
            Keyword overrides merged over the preset's kwargs (e.g. ``max_length``).

        Returns
        -------
        CFGGenerator
        """
        from symseq.generators.presets import cfg as cfg_presets

        if preset_name not in cfg_presets.PRESETS:
            raise ValueError(f"CFG preset {preset_name!r} not found. Available: {cfg_presets.list_presets()}")
        preset = dict(cfg_presets.__dict__[preset_name])
        preset.update(overrides)
        return cls(seed=seed, **preset)

    # ============================ Grammar construction ============================

    @staticmethod
    def _build_grammar(grammar: "str | PCFG | CFG") -> "PCFG":
        """Coerce the ``grammar`` argument into an ``nltk.PCFG``."""
        if isinstance(grammar, PCFG):
            return grammar
        if isinstance(grammar, CFG):
            return CFGGenerator._cfg_to_pcfg(grammar)

        if isinstance(grammar, (str, Path)):
            text = CFGGenerator._maybe_read_file(str(grammar))
            # NLTK infers the start symbol from the first production's LHS.
            try:
                return PCFG.fromstring(text)
            except ValueError:
                # No probabilities supplied: parse as a plain CFG, then uniformize.
                cfg = CFG.fromstring(text)
                return CFGGenerator._cfg_to_pcfg(cfg)

        raise TypeError(f"`grammar` must be a str, Path, nltk.CFG, or nltk.PCFG; got {type(grammar).__name__}")

    @staticmethod
    def _maybe_read_file(grammar: str) -> str:
        """Return file contents if ``grammar`` points to an existing file, else itself."""
        # A grammar string contains "->"; a path will not. Guard the Path() call so
        # multi-line grammar strings don't trip the filesystem check.
        if "->" not in grammar:
            path = Path(grammar)
            if path.exists():
                return path.read_text()
        return grammar

    @staticmethod
    def _cfg_to_pcfg(cfg: "CFG") -> "PCFG":
        """Assign uniform probabilities to productions sharing a left-hand side."""
        by_lhs: dict = {}
        for prod in cfg.productions():
            by_lhs.setdefault(prod.lhs(), []).append(prod)

        prob_prods = []
        for lhs, prods in by_lhs.items():
            p = 1.0 / len(prods)
            for prod in prods:
                prob_prods.append(ProbabilisticProduction(lhs, prod.rhs(), prob=p))
        return PCFG(cfg.start(), prob_prods)

    @staticmethod
    def _extract_terminals(grammar: "PCFG") -> list[str]:
        """Collect the set of terminal symbols emitted by the grammar."""
        terminals = set()
        for prod in grammar.productions():
            for sym in prod.rhs():
                if not isinstance(sym, Nonterminal):
                    terminals.add(sym)
        return sorted(terminals)

    # ================================ Generation =================================

    def generate_string(self, max_attempts: int | None = None) -> list[str]:
        """
        Sample a single grammatical string via stochastic leftmost derivation.

        Productions are expanded by sampling among those sharing the leftmost
        nonterminal, weighted by their probabilities. Derivations that exceed
        ``max_length``/``max_depth`` are rejected and resampled.

        Parameters
        ----------
        max_attempts : int, optional
            Override the instance ``max_attempts`` for this call.

        Returns
        -------
        list of str
            A terminal string in the language of the grammar.

        Raises
        ------
        RuntimeError
            If no string within the length/depth bounds is found in ``max_attempts``.
        """
        attempts = max_attempts if max_attempts is not None else self.max_attempts
        for _ in range(attempts):
            result = self._derive()
            if result is not None:
                return result
        raise RuntimeError(
            f"Failed to sample a string within max_length={self.max_length}, "
            f"max_depth={self.max_depth} after {attempts} attempts. "
            f"Consider relaxing these bounds or the grammar's recursion."
        )

    def _derive(self) -> list[str] | None:
        """One leftmost-derivation attempt; returns terminals or None if out of bounds."""
        # Frontier of symbols still to expand, processed left to right.
        frontier: list = [self._start]
        terminals: list[str] = []
        steps = 0

        while frontier:
            sym = frontier.pop(0)
            if not isinstance(sym, Nonterminal):
                terminals.append(sym)
                if len(terminals) > self.max_length:
                    return None
                continue

            steps += 1
            if steps > self.max_depth:
                return None

            prods = self._productions_by_lhs.get(sym)
            if not prods:
                # Nonterminal with no production: dead derivation.
                return None
            prod = self._sample_production(prods)
            frontier = list(prod.rhs()) + frontier

        if len(terminals) > self.max_length:
            return None
        return terminals

    def _sample_production(self, prods: list) -> "ProbabilisticProduction":
        """Pick a production weighted by probability using the instance RNG."""
        if len(prods) == 1:
            return prods[0]
        weights = np.array([p.prob() for p in prods], dtype=float)
        weights /= weights.sum()
        idx = int(self.rng.choice(len(prods), p=weights))
        return prods[idx]

    def generate_string_set(self, n: int) -> list[list[str]]:
        """Generate ``n`` grammatical strings."""
        return [self.generate_string() for _ in range(n)]

    # ============================ Grammaticality oracle ===========================

    def is_grammatical(self, string: Sequence[str]) -> bool:
        """
        Return True if ``string`` is in the language of the grammar.

        Uses NLTK's chart parser (the same machinery as
        :func:`symseq.metrics.grammar.inference.cyk_parse`) over the generator's
        own grammar, so the grammar is both producer and oracle.
        """
        tokens = list(string)
        if not tokens:
            return False
        # Any token outside the grammar's terminals cannot parse.
        if any(tok not in self.alphabet for tok in tokens):
            return False
        try:
            return any(True for _ in self._parser.parse(tokens))
        except ValueError:
            # Token unknown to the grammar.
            return False

    # =============================== Negative samples ==============================

    def generate_nongrammatical_strings(
        self,
        n: int,
        n_deviants: int = 1,
        strategies: Sequence[str] | None = None,
        max_attempts: int | None = None,
        verify_illegal: bool = True,
    ) -> list[list[str]]:
        """
        Generate illegal strings by locally corrupting grammatical ones.

        Parameters
        ----------
        n : int
            Number of illegal strings to generate.
        n_deviants : int, optional
            Number of corruption operations per string. Default ``1``.
        strategies : sequence of {{'replace_symbol','insert','delete','swap_adjacent','truncate'}}, optional
            Allowed corruption strategies. If ``None``, all are used.
        max_attempts : int, optional
            Maximum attempts per string. Defaults to the instance ``max_attempts``.
        verify_illegal : bool, optional
            If ``True``, verify each corrupted string fails to parse and retry
            otherwise. Default ``True``.

        Returns
        -------
        list of list of str
            Strings that are *not* in the language of the grammar.

        Raises
        ------
        RuntimeError
            If an illegal string cannot be produced within ``max_attempts``.
        """
        if strategies is None:
            strategies = ("replace_symbol", "insert", "delete", "swap_adjacent", "truncate")
        attempts = max_attempts if max_attempts is not None else self.max_attempts

        out: list[list[str]] = []
        for _ in range(n):
            for _attempt in range(attempts):
                string = self.generate_string()[:]
                for _k in range(max(0, n_deviants)):
                    strat = self._choice(strategies)
                    string = self._apply_violation(string, strat)

                if not verify_illegal or not self.is_grammatical(string):
                    out.append(string)
                    break
            else:
                raise RuntimeError(
                    f"Failed to generate an illegal string after {attempts} attempts. "
                    f"Try increasing n_deviants or max_attempts."
                )
        return out

    def _apply_violation(self, string: list[str], strategy: str) -> list[str]:
        """Apply a single corruption strategy to a token list."""
        if not string:
            return string[:]
        t = string[:]
        alphabet = self.alphabet

        if strategy == "replace_symbol":
            idx = int(self.rng.integers(0, len(t)))
            alternatives = [s for s in alphabet if s != t[idx]]
            if alternatives:
                t[idx] = self._choice(alternatives)
            return t

        if strategy == "insert":
            pos = int(self.rng.integers(0, len(t) + 1))
            t.insert(pos, self._choice(alphabet))
            return t

        if strategy == "delete":
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

    # ================================ Trial API ===================================

    def generate_trial(
        self,
        grammatical: bool = True,
        n_deviants: int = 1,
        strategies: Sequence[str] | None = None,
        max_attempts: int | None = None,
        **kwargs,
    ) -> Trial:
        """
        Generate one Trial.

        ``Trial.intrinsic_targets["grammaticality"]`` is a per-trial bool —
        ``True`` for a string sampled from the grammar, ``False`` for one
        produced by the corruption procedure.
        """
        if grammatical:
            symbols = self.generate_string(max_attempts=max_attempts)
            is_gram = True
        else:
            symbols = self.generate_nongrammatical_strings(
                n=1,
                n_deviants=n_deviants,
                strategies=strategies,
                max_attempts=max_attempts,
            )[0]
            is_gram = False

        intrinsic_targets = {
            "grammaticality": Target(values=is_gram, mask=None, granularity="per_trial"),
        }
        meta = {
            "paradigm": "CFG",
            "grammar": self.label,
            "length": len(symbols),
        }
        return Trial(symbols=symbols, meta=meta, intrinsic_targets=intrinsic_targets)

    # ================================= Utilities ==================================

    def _choice(self, seq: Sequence):
        """RNG-friendly choice over an arbitrary Python sequence."""
        return seq[int(self.rng.integers(0, len(seq)))]

    def __repr__(self) -> str:
        return (
            f"CFGGenerator(start={self._start!r}, "
            f"n_productions={len(self.grammar.productions())}, "
            f"alphabet_size={self.alphabet_size}, "
            f"max_length={self.max_length}, max_depth={self.max_depth})"
        )
