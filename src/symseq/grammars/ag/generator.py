# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""
generator.py

Util functions to randomly generate grammars that satisfy certain constraints.
"""

# TODO consider sampling-based generation

import numpy as np
import random
import networkx as nx
from sklearn.preprocessing import normalize
import copy

from symseq.core import state
from symseq.core.state import State
from symseq.grammars.ag.synthesis import generate_grammar_with_target_te_complexity
from symseq.utils.io import get_logger

logger = get_logger(__name__)

default_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def _validate_parameters(alphabet_size, ambiguities, ambiguity_depth, n_terminal_states):
    # some conditions for which we certainly cannot / should not generate valid grammars
    if ambiguities > 0 and ambiguity_depth == 0:
        raise ValueError("Could not generate grammar for because ambiguities > 0 and ambiguity_depth = 0!")
    if ambiguities > alphabet_size:
        raise ValueError("Number of ambiguities cannot be greater than the number of symbols in the alphabet.")
    if n_terminal_states == 0:
        raise ValueError("Number of terminal states cannot be 0.")

    if alphabet_size > len(default_alphabet):
        raise ValueError(
            f"Alphabet size cannot be greater than the number of symbols in the default alphabet: {len(default_alphabet)}"
        )


def generate_random_grammar(
    label: str = "Random artificial grammar",
    alphabet_size: int = 4,
    ambiguities: int = 0,
    ambiguity_depth: int = 0,
    n_start_states: int = 1,
    n_terminal_states: int = 1,
    transition_density: float = 0.25,
    assume_equiprobable: bool = True,
    min_string_length: int = 1,
    max_iter: int = int(1e5),
    eos: str = "#",
    rng: np.random.Generator | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Generate the parameter dictionary for an artificial grammar following the specified constraints and properties. Formally, each grammatical (valid) string will end with the EOS symbol.

    Parameters
    ----------
    label : str
        Grammar label.
    alphabet_size : int
        Size of the alphabet or number of symbols.
    ambiguities : int
        How many symbols can be repeated.
    ambiguity_depth : int
        Specifies the number of different instances of each repeatable symbol. Only relevant if ambiguities > 0.
    start_states : int
        Number of initial states.
    terminal_states : int
        Number of terminal states corresponding to regular symbols (not EOS), with direct transitions to the terminal state (EOS).
    transition_density : float
        Density of the transition matrix.
    assume_equiprobable : bool
        Assume equiprobable outgoing transitions from each state.
    min_string_length : int
        Minimum length of valid strings in the grammar.
    max_tries : int
        Maximum number of attempts to generate a valid grammar.
    rng : np.random.RandomState or None
        Random number generator instance for reproducibility. If None, a new default generator is created with seed `seed`.
    seed : int
        Random seed for reproducibility, default is 42. This parameter is ignored if `rng` is not None.
    verbose : bool
        Display progress information.

    Returns
    -------
    dict
        Dictionary with grammar properties, such as states and transition table.

    Notes
    -----
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    random.seed(int(rng.integers(0, 100)))

    _validate_parameters(alphabet_size, ambiguities, ambiguity_depth, n_terminal_states)
    valid_grammar = False

    # generate grammar parameters
    alphabet = list(default_alphabet[:alphabet_size])  # list of symbols

    cnt = 0
    while valid_grammar is False and cnt < max_iter:
        # list of states (possibly indexed symbols). NOTE: here we use Python str type for states, not core.state.State
        states = alphabet.copy()

        if ambiguities:
            repeatable_symbols = list(random.sample(alphabet, ambiguities))  # which symbols can repeat

            # how many repetitions?
            for symb in repeatable_symbols:
                states.remove(symb)
                # replace each repeatable symbol with indexed states
                new_states = [State(symb, index=i).tostring() for i in range(ambiguity_depth)]
                states.extend(new_states)

        states_no_eos = copy.deepcopy(states)
        states.append(eos)  # EOS belongs to the states but not to the alphabet

        # choose start states randomly (except for EOS)
        start_states = list(rng.choice(states_no_eos, n_start_states, replace=False).astype(object))
        # select terminal states
        terminal_states = list(rng.choice(states_no_eos, n_terminal_states, replace=False).astype(object))
        assert states[-1] == eos, "EOS should have been the last added state."

        G = init_graph(states_no_eos, start_states, terminal_states, transition_density, rng)
        if G is None:
            cnt += 1
            continue

        # add transitions from terminal states to the EOS
        G.add_node(eos)
        for s in terminal_states:
            G.add_edge(s, eos)

        # fill transition table with graph edges (source X target)
        A = np.zeros((len(states), len(states)))  # adjacency matrix
        state_to_idx = {s: i for i, s in enumerate(states)}
        for u, v in G.edges:
            A[state_to_idx[u], state_to_idx[v]] = 1

        # ensure that there are no states with zero input transitions (every state must be reachable)
        assert np.all(np.sum(A, 0))

        # assume equiprobable transitions
        if assume_equiprobable:
            A = normalize(A, axis=1, norm="l1")
        else:
            raise NotImplementedError("Non-equiprobable transitions not implemented yet.")
            A = np.asarray(A * t.todense())
            A = normalize(A, axis=1, norm="l1")

        # store transitions
        transitions = []
        for x, y in zip(A.nonzero()[0], A.nonzero()[1]):
            transitions.append((states[x], states[y], A[x, y]))
            G[states[x]][states[y]]["weight"] = np.round(A[x, y], decimals=2)

        valid_grammar = _check_grammar_validity(G, start_states, min_string_length, eos, verbose=False)

        cnt += 1  # increase the attempt counter

    if valid_grammar is False:
        raise RuntimeError(f"Could not generate grammar for the specified parameters after {max_iter} attempts!")

    return {
        "label": label,
        "states": states,
        "alphabet": alphabet,
        "start_states": start_states,
        "terminal_states": terminal_states,
        "transitions": transitions,
        "eos": eos,
    }


def init_graph(states_no_eos, start_states, terminal_states, transition_density, rng):
    """
    Generate a random sparse but strongly connected directed graph.

    Parameters
    ----------
    states_no_eos : list of hashable
        List of all states excluding the EOS (end-of-sequence) state.
    start_states : list of hashable
        Subset of states that can serve as valid starting states.
    terminal_states : list of hashable
        Subset of states that can serve as valid terminal states.
    transition_density : float
        Fraction of possible directed edges to include in the graph.
        Must satisfy `n / (n^2) <= transition_density <= (n-1) / n`,
        where `n = len(states_no_eos)`.
    rng : numpy.random.Generator
        Random number generator instance used for reproducibility.

    Returns
    -------
    G : nx.DiGraph or None
        A strongly connected directed graph with `len(states_no_eos)` nodes
        and approximately `n^2 * transition_density` edges.
        Returns `None` if a randomly chosen start state coincides with
        the chosen terminal state (retry recommended).

    Raises
    ------
    ValueError
        If the requested number of edges is less than `n`, making strong
        connectivity impossible.
    ValueError
        If the requested number of edges exceeds `n * (n - 1)`, the maximum
        possible number of edges without self-loops.
    """
    n = len(states_no_eos)
    m = int(np.ceil(n * n * transition_density))

    if m < n:
        # raise ValueError("Need at least n edges for strong connectivity.")
        raise RuntimeError(
            f"Could not generate a valid transition matrix (strongly connected graph)"
            f"with the provided density (too low?)."
        )
    if m > n * (n - 1):
        raise ValueError("Too many edges: maximum is n*(n-1).")

    G = nx.DiGraph()
    G.add_nodes_from(states_no_eos)

    # ensure strong connectivity by creating a random cycle
    cycle_nodes = copy.deepcopy(states_no_eos)
    rng.shuffle(cycle_nodes)

    # Ensure cycle is between a start and terminal state!
    # pick one starting node randomly and add it to the front of the cycle
    start_node = start_states[rng.integers(0, len(start_states))]
    idx = cycle_nodes.index(start_node)
    cycle_nodes[0], cycle_nodes[idx] = cycle_nodes[idx], cycle_nodes[0]
    # pick one terminal node randomly and add it to the back of the cycle
    terminal_node = terminal_states[rng.integers(0, len(terminal_states))]
    idx = cycle_nodes.index(terminal_node)
    cycle_nodes[-1], cycle_nodes[idx] = cycle_nodes[idx], cycle_nodes[-1]
    if start_node == terminal_node:
        return None  # break here and let the outer loop try again

    cycle_edges = np.column_stack([cycle_nodes, np.roll(np.array(cycle_nodes, dtype=object), -1)])
    G.add_edges_from(cycle_edges)

    # build all possible edges (excluding self-loops)
    all_edges = np.array([[u, v] for u in states_no_eos for v in states_no_eos], dtype=object)

    # exclude cycle edges efficiently: convert cycle edges to a set of tuples for fast membership test
    cycle_set = {tuple(e) for e in cycle_edges}
    mask = np.array([tuple(e) not in cycle_set for e in all_edges], dtype=bool)
    available_edges = all_edges[mask]

    # sample remaining edges
    extra_edges = available_edges[rng.choice(available_edges.shape[0], size=m - n, replace=False)]

    G.add_edges_from(extra_edges)

    return G


def _check_grammar_validity(
    G: nx.Graph,
    start_states: list[str],
    min_string_length: int,
    eos: str,
    verbose: bool,
) -> bool:
    """
    Check if the grammar adheres to specified constraints.

    This function validates the following constraints:
    - For each start state, there should be a path of valid length to a terminal state.
    - (Optional) For all states, there should be a path to a terminal state (no absorbing states).
    - (Optional) All states must be reachable from start states, and each should have a
      path to a terminal state.

    Parameters
    ----------
    G : nx.Graph
        The graph representing the grammar.
    start_states : list of str
        Initial states.
    min_string_length : int
        Minimum length required for a valid path.
    eos : str
        End-of-string marker.
    verbose : bool
        If True, logs additional information about path validation.

    Returns
    -------
    bool
        True if the grammar is valid, False otherwise.
    """
    path_exists = False
    states = start_states

    # if no dead states or disconnected states are allowed, check paths from all states not just the initial ones
    states = list(G.nodes)

    for s in states:
        try:
            path_exists = nx.shortest_path(G, source=s, target=eos)

            if s in start_states and len(path_exists) - 1 < min_string_length:
                if verbose:
                    logger.info(
                        f"Shortest path from start state {s} to end state {eos} is below minimum expected length"
                    )
                path_exists = False
                break
        except nx.NetworkXNoPath:
            if verbose:
                logger.info(f"No path from start state {s} to end state {eos}")
            path_exists = False
            break

    # to avoid unreachable cycles, it's enough to ensure all nodes are reachable from the initial states
    if path_exists:
        for tgt in list(G.nodes):
            if tgt not in start_states:
                reachable = False
                for start in start_states:
                    try:
                        nx.shortest_path(G, source=start, target=tgt)
                        reachable = True
                        break
                    except nx.NetworkXNoPath:
                        continue
                if not reachable:
                    if verbose:
                        logger.info(f"No path from an initial state to {tgt}")
                    return False

    return path_exists is not False


def _consistency_check_complexity_dataset(df, kwargs):
    """
    Check if the provided kwargs are consistent with the dataset.

    Parameters
    ----------
    kwargs : dict
        Dictionary of keyword arguments.

    Raises
    ------
    ValueError
        If any of the provided arguments are inconsistent with the dataset.
    """
    if kwargs["TE_range"][0] > kwargs["TE_range"][1]:
        raise ValueError("TE_range[0] must be less than or equal to TE_range[1].")

    params_range_check = [
        "TE_range",
        "alphabet_size",
        "ambiguities",
        "ambiguity_depth",
        "n_start_states",
        "transition_density",
    ]
    for param in params_range_check:
        df_param = param if param != "TE_range" else "TE_direct"  # legacy reasons
        if kwargs[param][0] < df[df_param].min() or kwargs[param][1] > df[df_param].max():
            raise ValueError(
                f"{param} is outside the range of the dataset: {kwargs[param]} vs ({df[df_param].min()}, {df[df_param].max()})"
            )


def grammar_with_complexity(
    target_complexity: float,
    label: str = "Random artificial grammar",
    alphabet_size: int = 4,
    ambiguities: int = 0,
    ambiguity_depth: int = 0,
    n_start_states: int = 1,
    n_terminal_states: int = 1,
    transition_density=None,
    assume_equiprobable: bool = True,
    min_string_length: int = 1,
    max_iter: int = int(1e5),
    eos: str = "#",
    rng: np.random.Generator | None = None,
    seed: int = 42,
    verbose: bool = True,
    **synthesis_kwargs,
):
    """
    Thin wrapper for `symseq.grammars.ag.synthesis.generate_grammar` to generate a grammar with a target TE complexity.
    """
    if transition_density and "p_mean" in synthesis_kwargs:
        if transition_density != synthesis_kwargs["p_mean"]:
            raise ValueError(
                "Transition density and p_mean cannot be specified simultaneously, or they must be the same."
            )
    elif transition_density is not None:
        synthesis_kwargs["p_mean"] = transition_density

    if not assume_equiprobable:
        raise NotImplementedError("Non-equiprobable transitions not implemented yet for this generator.")

    if rng is None:
        rng = np.random.default_rng(seed)

    random.seed(int(rng.integers(0, 100)))
    n_nodes = (alphabet_size - ambiguities) + ambiguity_depth * ambiguities  # EOS not included

    # adjMat is src x tgt
    adj_mat, counts, complexities, densities, scores, step_ids, change, t_tot, t_it = (
        generate_grammar_with_target_te_complexity(
            target_complexity=target_complexity,
            n_nodes=n_nodes,
            **synthesis_kwargs,
        )
    )

    # construct the grammar parameter dictionary
    alphabet = list(default_alphabet[:alphabet_size])  # list of symbols
    states = alphabet.copy()
    if ambiguities:
        repeatable_symbols = list(random.sample(alphabet, ambiguities))  # which symbols can repeat

        # how many repetitions?
        for symb in repeatable_symbols:
            states.remove(symb)
            # replace each repeatable symbol with indexed states
            new_states = [State(symb, index=i).tostring() for i in range(ambiguity_depth)]
            states.extend(new_states)

    states_no_eos = copy.deepcopy(states)
    states.append(eos)  # EOS belongs to the states but not to the alphabet
    # choose start states randomly (except for EOS)
    start_states = list(rng.choice(states_no_eos, n_start_states, replace=False).astype(object))
    # select terminal states
    terminal_states = list(rng.choice(states_no_eos, n_terminal_states, replace=False).astype(object))

    # add zeros for EOS as last row and column
    adj_mat = np.pad(adj_mat, ((0, 1), (0, 1)), mode="constant", constant_values=0)
    # add transitions from EOS to start states
    for s in start_states:
        adj_mat[-1, states.index(s)] = 1
    # add transitions from terminal states to the EOS
    for s in terminal_states:
        adj_mat[states.index(s), -1] = 1

    # assume equiprobable transitions
    A = normalize(adj_mat, axis=1, norm="l1")

    # store transitions
    transitions = []
    for x, y in zip(A.nonzero()[0], A.nonzero()[1]):
        transitions.append((states[x], states[y], A[x, y]))

    grammar_params = {
        "label": label,
        "states": states,
        "alphabet": alphabet,
        "start_states": start_states,
        "terminal_states": terminal_states,
        "transitions": transitions,
        "eos": eos,
    }

    metadata = {
        "counts": counts,
        "complexities": complexities,
        "densities": densities,
        "scores": scores,
        "step_ids": step_ids,
        "change": change,
        "t_tot": t_tot,
        "t_it": t_it,
    }
    return grammar_params, metadata
