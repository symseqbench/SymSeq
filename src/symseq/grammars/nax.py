# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""
Models n-AX (conditional one-back) as a regular grammar.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from symseq.grammars.ag import ArtificialGrammar
from symseq.utils.io import get_logger

logging = get_logger(__name__)


class nAX(ArtificialGrammar):
    """
    n-AX (conditional one-back) as a regular grammar with a SymSeq-friendly sampler.

    A trial is a short symbolic sequence:
        [ Context_i, Cue_i, (0..k Fillers), Probe_j ]

    Target rule per context i:
        Probe must equal Probe_i (j == i). Any j != i is a lure.

    Parameters
    ----------
    label : str
        Name for bookkeeping. Default "n-AX".
    contexts : sequence of str
        Context symbols, length n (e.g., ('1','2','3')). Default ("1", "2").
    cue_map : dict[str, str], optional
        Context -> cue symbol for each context. Default {"1": "A", "2": "B"}.
    probe_map : dict[str, str], optional
        Context -> probe (target) symbol for each context. Default {"1": "X", "2": "Y"}.
    fillers : sequence of str
        Symbols that can occur between cue and probe. Default ("C", "D", "Z").
    context_probs : sequence of float, optional
        Probability of sampling each context as the trial's context. Defaults to uniform.
    probe_given_context : np.ndarray, shape (n, n), optional
        Row-stochastic matrix where row i gives P(probe_ctx=j | context=i).
        If None, `p_target` is used with uniform lures among j!=i.
    p_target : float, optional
        Convenience for building `probe_given_context` if that matrix is not provided.
        For each row i: P(j=i) = p_target; P(j≠i) equally share (1 - p_target). Default 0.6.
    eos : str
        End-of-sequence symbol used by ArtificialGrammar. Default "#".
    rng : numpy.random.Generator, optional
        RNG to use. If None, a new default generator is created.
    seed : int
        Seed used only if rng is None. Default 42.
    verbose : bool
        Verbose initialization of the base grammar. Default False.

    Notes
    -----
    - All target and lure trials are grammatical in the underlying grammar.
      The distinction is in labeling (i.e., j==i is target).
    - The grammar keeps filler states context-specific to retain context memory until the probe.
    """

    def __init__(
        self,
        label: str = "n-AX",
        contexts: Sequence[str] = ("1", "2"),
        cue_map: Optional[Dict[str, str]] = None,
        probe_map: Optional[Dict[str, str]] = None,
        fillers: Sequence[str] = ("C", "D", "Z"),
        context_probs: Optional[Sequence[float]] = None,
        probe_given_context: Optional[np.ndarray] = None,
        p_target: float = 0.6,
        eos: str = "#",
        rng: Optional[np.random.Generator] = None,
        seed: int = 42,
        verbose: bool = False,
    ):
        self.label = label
        self.contexts = contexts
        self.cue_map = cue_map if cue_map is not None else {"1": "A", "2": "B"}
        self.probe_map = probe_map if probe_map is not None else {"1": "X", "2": "Y"}
        self.fillers = fillers
        self.context_probs = context_probs
        self.probe_given_context = probe_given_context
        self.p_target = p_target
        self.eos = eos
        self.seed = seed
        self.verbose = verbose

        # Setup RNG
        self.rng = rng if rng is not None else np.random.default_rng(self.seed)

        # Validate and setup
        self._validate_contexts()
        self._setup_context_index()
        self._setup_context_probs()
        self._setup_probe_matrix()

        # Build grammar components
        alphabet, states, start_states, terminal_states, transitions = self._build_grammar()

        # Initialize base class
        super().__init__(
            label=self.label,
            states=states,
            alphabet=alphabet,
            transitions=transitions,
            start_states=start_states,
            terminal_states=terminal_states,
            eos=self.eos,
            rng=self.rng,
            seed=self.seed,
            validate=True,
            build_graph=True,
            verbose=self.verbose,
        )

    def _validate_contexts(self) -> None:
        """
        Validate that contexts, cue_map, and probe_map are properly configured.
        """
        n = len(self.contexts)
        if n < 2:
            raise ValueError("Provide at least 2 contexts for n-AX.")
        if set(self.contexts) - set(self.cue_map.keys()):
            raise ValueError("cue_map must define a cue for each context.")
        if set(self.contexts) - set(self.probe_map.keys()):
            raise ValueError("probe_map must define a probe for each context.")

    def _setup_context_index(self) -> None:
        """
        Build internal index mapping from context symbols to indices.
        """
        self._ctx_index = {c: i for i, c in enumerate(self.contexts)}

    def _setup_context_probs(self) -> None:
        """
        Setup context probabilities. If not provided, defaults to uniform.
        """
        n = len(self.contexts)
        if self.context_probs is None:
            self.context_probs = np.full(n, 1.0 / n, dtype=float)
        else:
            self.context_probs = np.array(self.context_probs, dtype=float)
            if len(self.context_probs) != n or np.any(self.context_probs < 0):
                raise ValueError("context_probs must be length-n and nonnegative.")
            self.context_probs = self.context_probs / self.context_probs.sum()

    def _setup_probe_matrix(self) -> None:
        """
        Setup probe_given_context matrix. If not provided, builds from p_target.
        """
        n = len(self.contexts)
        if self.probe_given_context is None:
            if not (0.0 <= self.p_target <= 1.0):
                raise ValueError("p_target must be in [0,1].")
            M = np.empty((n, n), dtype=float)
            for i in range(n):
                M[i, :] = (1.0 - self.p_target) / (n - 1)
                M[i, i] = self.p_target
            self.probe_given_context = M
        else:
            raise NotImplementedError("Custom probe_given_context matrix not yet implemented.")
            # Future implementation would go here:
            # self.probe_given_context = np.array(self.probe_given_context, dtype=float)
            # if self.probe_given_context.shape != (n, n):
            #     raise ValueError("probe_given_context must have shape (n, n).")
            # if np.any(self.probe_given_context < 0):
            #     raise ValueError("probe_given_context must be nonnegative.")
            # row_sums = self.probe_given_context.sum(axis=1, keepdims=True)
            # if np.any(row_sums == 0):
            #     raise ValueError("Rows of probe_given_context must sum to > 0.")
            # self.probe_given_context = self.probe_given_context / row_sums

    def _build_grammar(self) -> Tuple[List[str], List[str], List[str], List[str], List[Tuple[str, str, float]]]:
        """
        Build alphabet, states, start states, terminal states, and transitions.

        Returns
        -------
        tuple
            (alphabet, states, start_states, terminal_states, transitions)
        """
        cues = set(self.cue_map.values())
        probes = set(self.probe_map.values())

        # Build alphabet
        alphabet = sorted(set(self.contexts) | cues | probes | set(self.fillers))

        # Build states
        states: List[str] = []
        start_states: List[str] = list(self.contexts)
        terminal_states: List[str] = list(probes)

        # Basic symbol states
        states.extend(list(set(self.contexts)))
        states.extend(list(cues))
        states.extend(list(probes))

        # Context-specific filler states: e.g., 'C(1)', 'C(2)', 'D(1)', ...
        for ctx in self.contexts:
            for f in self.fillers:
                states.append(f"{f}({ctx})")

        # EOS
        states.append(self.eos)

        # Build transitions
        transitions = self._build_transitions(cues, probes)

        return alphabet, states, start_states, terminal_states, transitions

    def _build_transitions(self, cues: set, probes: set) -> List[Tuple[str, str, float]]:
        """
        Build all state transitions for the grammar.

        Parameters
        ----------
        cues : set
            Set of cue symbols.
        probes : set
            Set of probe symbols.

        Returns
        -------
        list of tuple
            List of (source_state, target_state, probability) transitions.
        """
        transitions: List[Tuple[str, str, float]] = []

        # Context -> Cue (deterministic)
        for ctx in self.contexts:
            transitions.append((ctx, self.cue_map[ctx], 1.0))

        # Cue -> Fillers or Probes
        for ctx in self.contexts:
            self._add_cue_fanout(transitions, ctx, self.cue_map[ctx], probes)

        # Filler -> Filler or Probe
        for ctx in self.contexts:
            self._add_filler_fanout(transitions, ctx, probes)

        # Probes -> EOS
        for p in probes:
            transitions.append((p, self.eos, 1.0))

        return transitions

    def _add_cue_fanout(
        self, transitions: List[Tuple[str, str, float]], ctx: str, cue: str, probes: set
    ) -> None:
        """
        Add transitions from a cue state to filler and probe states.

        Parameters
        ----------
        transitions : list
            List to append transitions to.
        ctx : str
            Context symbol.
        cue : str
            Cue symbol for this context.
        probes : set
            Set of all probe symbols.
        """
        fan = []
        # Can go to any filler(ctx)
        for f in self.fillers:
            fan.append((cue, f"{f}({ctx})", 1.0))
        # Can go to any probe (both targets and lures are grammatical)
        for p in probes:
            fan.append((cue, p, 1.0))

        w = 1.0 / len(fan) if fan else 1.0
        for src, tgt, _ in fan:
            transitions.append((src, tgt, w))

    def _add_filler_fanout(self, transitions: List[Tuple[str, str, float]], ctx: str, probes: set) -> None:
        """
        Add transitions from filler states to other fillers or probes.

        Parameters
        ----------
        transitions : list
            List to append transitions to.
        ctx : str
            Context symbol.
        probes : set
            Set of all probe symbols.
        """
        for f in self.fillers:
            src = f"{f}({ctx})"
            fan = []
            # Can go to any filler(ctx)
            for g in self.fillers:
                fan.append((src, f"{g}({ctx})", 1.0))
            # Can go to any probe
            for p in probes:
                fan.append((src, p, 1.0))

            w = 1.0 / len(fan)
            for s, t, _ in fan:
                transitions.append((s, t, w))

    # ========================= High-level sampling =========================

    def generate_string(self, *args, **kwargs) -> list[str]:
        """
        Sample one n-AX trial: [Context_i, Cue_i, (Fillers...), Probe_j].
        Target iff i == j.

        Parameters
        ----------
        n_fillers : int, optional (keyword-only)
            Number of fillers to insert between cue and probe. Default is None. Extracted from kwargs.
        kwargs : dict
            Keyword arguments to pass to `generate_string`.

        Returns
        -------
        list of str
            A valid trial string, lure or target.
        """
        n = len(self.contexts)

        # choose context index i
        i = int(self.rng.choice(n, p=self.context_probs))
        ctx_i = self.contexts[i]
        cue_i = self.cue_map[ctx_i]

        max_tries = 1000
        n_fillers = kwargs.pop("n_fillers", None)
        length = 3 + n_fillers if n_fillers is not None else None
        kwargs["length_range"] = (length, length) if length is not None else None
        string = None
        while max_tries > 0:
            # choose probe context j given i (from n×n matrix)
            j = int(self.rng.choice(n, p=self.probe_given_context[i]))
            probe_j = self.probe_map[self.contexts[j]]

            max_tries_in = 100
            while max_tries_in > 0:
                string = super().generate_string(**kwargs)
                if string[-1] == probe_j:
                    break
                max_tries_in -= 1

            if max_tries_in == 0:
                max_tries -= 1
            else:
                break

        if max_tries == 0:
            raise RuntimeError("Could not generate a valid trial string.")

        # sanity: the underlying grammar should accept string + [eos]
        assert self.is_grammatical(string + [self.eos]), "Internal grammar mismatch."
        return string

    # TODO match signature of parent class
    def generate_string_set(self, n_samples: int = 1, **kwargs) -> list[list[str]]:
        """
        Create a set of n_samples trial strings.

        Parameters
        ----------
        n_samples : int, optional
            Number of trials to generate. Default is 1.
        kwargs : dict
            Keyword arguments to pass to `generate_string`.

        Returns
        -------
        list of list of str
            A list of trial strings.
        """
        # currently all generated strings are grammatical
        string_set = [self.generate_string(**kwargs) for _ in range(n_samples)]
        return string_set

    # ============================== Utilities ==============================
    # TODO This belongs more to the task definition
    def label_trial(self, string: List[str]) -> Tuple[str, bool]:
        """
        Label as target or lure and return (label, is_target).

        label format:
            'C{i}->T'          if probe matches context i
            'C{i}->L(j)'       if probe corresponds to context j != i
        """
        if len(string) < 3:
            return ("", False)

        ctx = string[0]
        cue = string[1]
        probe = string[-1]

        # validate cue belongs to ctx
        if cue != self.cue_map.get(ctx, None):
            return ("", False)

        # find which context the probe belongs to
        j = None
        for c in self.contexts:
            if probe == self.probe_map[c]:
                j = self._ctx_index[c]
                break
        if j is None:
            return ("", False)

        i = self._ctx_index[ctx]
        if i == j:
            return (f"C{i+1}->T", True)
        else:
            return (f"C{i+1}->L({j+1})", False)
