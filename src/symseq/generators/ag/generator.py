# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""
generator.py

Util functions to randomly generate grammars that satisfy certain constraints.
"""

# TODO consider sampling-based generation

import copy
import random
import warnings

import networkx as nx
import numpy as np

from symseq.core import state
from symseq.core.state import State
from symseq.generators.ag.synthesis import generate_grammar_with_target_te_complexity
from symseq.utils.io import get_logger

logger = get_logger(__name__)

default_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def _normalize_rows_l1(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)


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
    restricted: bool = False,
    graph_init: str = "tree",
    reject_equivalent_copies: bool = False,
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
    n_start_states : int
        Number of initial states.
    n_terminal_states : int
        Number of terminal states corresponding to regular symbols (not EOS), with direct transitions to the terminal state (EOS).
        Start and terminal states are drawn independently, so a state can be both. Such a state makes its own symbol a
        valid one-symbol string; use ``min_string_length >= 2`` to rule this out (every start state must then be at
        least two transitions away from EOS, which excludes start states that are also terminal).
    transition_density : float
        Density of the transition matrix.
    assume_equiprobable : bool
        Assume equiprobable outgoing transitions from each state.
    min_string_length : int
        Minimum length of valid strings in the grammar. Values of 2 or more also exclude grammars in which a start
        state is a terminal state.
    max_tries : int
        Maximum number of attempts to generate a valid grammar.
    rng : np.random.RandomState or None
        Random number generator instance for reproducibility. If None, a new default generator is created with seed `seed`.
    seed : int
        Random seed for reproducibility, default is 42. This parameter is ignored if `rng` is not None.
    verbose : bool
        Display progress information.
    restricted : bool
        Sample at most one outgoing target per observable symbol for each state. The restricted sampler is the default
        sampler conditioned on this rule: uniform over trim graphs with exactly ``ceil(n^2 * transition_density)``
        edges, with the same meaning of `transition_density` and the same edge count as in unrestricted mode (see
        :func:`init_restricted_graph`). A slot of a symbol with ``k`` copies can be filled in ``k`` ways, so such a
        symbol is the next symbol more often than a single-copy symbol, but less than ``k`` times as often, and each
        copy receives fewer in-edges than a single-copy state (the copies share one slot per source; the gap grows
        with density). Symbol-neutral sampling is not implemented. Only supported with
        ``graph_init="tree"``.
    graph_init : {"tree", "cycle"}
        Sampler for the underlying transition graph. ``"tree"`` (default) starts from a random spanning out-forest
        rooted at the start states and a random spanning in-forest rooted at the terminal states, so that grammars
        need not be strongly connected (start states may have no incoming transitions). ``"cycle"`` is the legacy
        sampler that plants a random Hamiltonian ring, always yielding strongly connected grammars; it is kept for
        reproducibility of earlier results and is incompatible with ``restricted=True``.
    reject_equivalent_copies : bool
        Reject (and retry, counting towards `max_iter`) grammars in which two states emitting the same symbol are
        bisimilar. Such copies generate the same future strings, so the nominal ambiguity has no effect. For restricted
        grammars (deterministic per symbol) the check is exact: copies are language-equivalent iff bisimilar. For
        non-restricted grammars it is only sufficient, so some language-equivalent copies are not rejected.
        Does not consume random numbers. Default False.

    Returns
    -------
    dict
        Dictionary with grammar properties, such as states and transition table.

    Raises
    ------
    ValueError
        If `graph_init` is not ``"tree"`` or ``"cycle"``, if ``restricted=True`` is combined with
        ``graph_init="cycle"``, or (tree sampler) if the requested density is too low to admit a trim grammar or
        requires more edges than available (``n^2``, or ``n * alphabet_size`` slots if ``restricted=True``).

    Warns
    -----
    UserWarning
        If ``graph_init="tree"`` (restricted or not) and the requested number of edges is below ``1.5 * n``
        (``n`` being the number of non-EOS states), where valid grammars are necessarily chain-like and the edge-swap
        chain mixes poorly.

    Notes
    -----
    A grammar is valid if every state is reachable from a start state, every state can reach EOS and the shortest
    string has at least `min_string_length` symbols. Strong connectivity is *not* required.
    """
    if graph_init not in ("tree", "cycle"):
        raise ValueError(f"graph_init must be 'tree' or 'cycle', got {graph_init!r}.")
    if restricted and graph_init == "cycle":
        raise ValueError("restricted=True is not supported with graph_init='cycle'; use graph_init='tree'.")

    if rng is None:
        rng = np.random.default_rng(seed)
    random.seed(int(rng.integers(0, 100)))

    _validate_parameters(alphabet_size, ambiguities, ambiguity_depth, n_terminal_states)
    valid_grammar = False

    if graph_init == "tree":
        n_states = alphabet_size - ambiguities + ambiguities * ambiguity_depth  # EOS not included
        n_edges = int(np.ceil(n_states * n_states * transition_density))
        if n_edges < 1.5 * n_states:
            warnings.warn(
                f"transition_density={transition_density} yields only {n_edges} edges for {n_states} states "
                "(< 1.5 * n_states). Near the minimum edge count, valid grammars are necessarily chain-like and "
                "the edge-swap sampler mixes poorly, so the sampled grammars are far from uniform.",
                UserWarning,
                stacklevel=2,
            )

    # generate grammar parameters
    alphabet = list(default_alphabet[:alphabet_size])  # list of symbols

    cnt = 0
    while valid_grammar is False and cnt < max_iter:
        # list of states (possibly indexed symbols). NOTE: here we use Python str type for states, not core.state.State
        states = alphabet.copy()
        states_rep = alphabet.copy()

        if ambiguities:
            repeatable_symbols = list(random.sample(alphabet, ambiguities))  # which symbols can repeat

            # how many repetitions?
            for symb in repeatable_symbols:
                index = states.index(symb)
                states.pop(index)
                states_rep.pop(index)
                # replace each repeatable symbol with indexed states
                new_states = [State(symb, index=i).tostring() for i in range(ambiguity_depth)]
                states.extend(new_states)
                states_rep.extend([symb] * len(new_states))

        states_no_eos = copy.deepcopy(states)
        states.append(eos)  # EOS belongs to the states but not to the alphabet

        # choose start states randomly (except for EOS)
        start_states = list(rng.choice(states_no_eos, n_start_states, replace=False).astype(object))
        # select terminal states
        terminal_states = list(rng.choice(states_no_eos, n_terminal_states, replace=False).astype(object))
        assert states[-1] == eos, "EOS should have been the last added state."

        if graph_init == "cycle":
            G = init_cycle_graph(states_no_eos, start_states, terminal_states, transition_density, rng)
        elif restricted:
            G = init_restricted_graph(
                states_no_eos, states_rep, alphabet, start_states, terminal_states, transition_density, rng
            )
        else:
            G = init_tree_graph(states_no_eos, start_states, terminal_states, transition_density, rng)
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

        # Start states may legitimately have no incoming transitions; every other state (including EOS) needs one.
        non_start_mask = np.ones(len(states), dtype=bool)
        non_start_mask[[state_to_idx[s] for s in start_states]] = False
        assert np.all(np.sum(A, 0)[non_start_mask] >= 1)

        # assume equiprobable transitions
        if assume_equiprobable:
            A = _normalize_rows_l1(A)
        else:
            raise NotImplementedError("Non-equiprobable transitions not implemented yet.")
            A = np.asarray(A * t.todense())
            A = _normalize_rows_l1(A)

        # store transitions
        transitions = []
        for x, y in zip(A.nonzero()[0], A.nonzero()[1]):
            transitions.append((states[x], states[y], A[x, y]))
            G[states[x]][states[y]]["weight"] = np.round(A[x, y], decimals=2)

        valid_grammar = _check_grammar_validity(G, start_states, min_string_length, eos, verbose=False)
        if valid_grammar and reject_equivalent_copies and _has_equivalent_copies(states, transitions, eos):
            valid_grammar = False

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


def _has_equivalent_copies(states, transitions, eos):
    """
    Check whether two states emitting the same symbol are bisimilar.

    Bisimilar copies generate the same future strings. The coarsest bisimulation is computed by partition refinement,
    starting from the partition by emitted symbol (EOS in its own class) and keying each state by its class and the set
    of classes of its successors until the number of classes stops changing.

    Parameters
    ----------
    states : list of str
        All states, EOS included. Indexed states are named ``"A(1)"``.
    transitions : iterable of tuple
        ``(source, target, probability)`` triples.
    eos : str
        End-of-sequence state.

    Returns
    -------
    bool
        True if two distinct states with the same symbol fall into the same class.
    """
    succ = {s: set() for s in states}
    for u, v, *_ in transitions:
        succ[u].add(v)
    class_id = {s: eos if s == eos else State.from_string(s).symbol for s in states}
    n_classes = len(set(class_id.values()))
    while True:
        key = {s: (class_id[s], frozenset(class_id[v] for v in succ[s])) for s in states}
        ids = {k: i for i, k in enumerate(dict.fromkeys(key.values()))}
        class_id = {s: ids[key[s]] for s in states}
        new_classes = len(ids)
        if new_classes == n_classes:
            break
        n_classes = new_classes
    return n_classes < len(states)


def _sample_rooted_forest(nodes, roots, rng):
    """
    Sample a uniformly random rooted spanning forest of the complete directed graph (no self-loops).

    Uses Wilson's algorithm (loop-erased random walks), where a step from node ``x`` goes to a node drawn uniformly
    from all other nodes.

    Parameters
    ----------
    nodes : sequence of hashable
        All nodes of the graph.
    roots : iterable of hashable
        Root nodes (non-empty subset of `nodes`).
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    dict
        ``next_hop[v]`` is the next node on the path from ``v`` to its root, for every non-root node ``v``. The
        sample is uniform over spanning forests in which every tree contains exactly one root.
    """
    nodes = list(nodes)
    n = len(nodes)
    index = {v: i for i, v in enumerate(nodes)}
    in_forest = [False] * n
    for r in roots:
        in_forest[index[r]] = True
    nxt = [-1] * n

    for i in range(n):
        u = i
        while not in_forest[u]:
            j = int(rng.integers(0, n - 1))  # uniform over the n - 1 nodes other than u
            nxt[u] = j + 1 if j >= u else j
            u = nxt[u]
        u = i
        while not in_forest[u]:  # add the loop-erased path to the forest
            in_forest[u] = True
            u = nxt[u]

    return {nodes[i]: nodes[nxt[i]] for i in range(n) if nxt[i] >= 0}


def _is_trim(succ, pred, start_mask, terminal_mask):
    """
    Check that every node is reachable from a start node and can reach a terminal node.

    Parameters
    ----------
    succ : list of int
        ``succ[u]`` is the bitmask of the successors of node ``u``.
    pred : list of int
        ``pred[v]`` is the bitmask of the predecessors of node ``v``.
    start_mask : int
        Bitmask of the start nodes.
    terminal_mask : int
        Bitmask of the terminal nodes.

    Returns
    -------
    bool
        True if the graph is trim.
    """
    full = (1 << len(succ)) - 1
    for adj, seed_mask in ((succ, start_mask), (pred, terminal_mask)):
        seen = seed_mask
        frontier = seed_mask
        while frontier:
            new = 0
            while frontier:
                low = frontier & -frontier
                new |= adj[low.bit_length() - 1]
                frontier ^= low
            frontier = new & ~seen
            seen |= new
        if seen != full:
            return False
    return True


def _min_trim_edges(n, start_states, terminal_states):
    """
    Minimum number of edges of a trim graph on ``n`` nodes.

    Parameters
    ----------
    n : int
        Number of nodes.
    start_states : sequence of hashable
        Start nodes.
    terminal_states : sequence of hashable
        Terminal nodes.

    Returns
    -------
    int
        ``max(n - |S|, n - |T|)``, plus one if ``S == T`` and ``S`` does not cover all nodes (the nodes outside ``S``
        then need a chain that leaves ``S`` and returns to it).
    """
    n_start = len(set(start_states))
    n_terminal = len(set(terminal_states))
    return max(n - n_start, n - n_terminal) + (1 if set(start_states) == set(terminal_states) and n > n_start else 0)


def _forest_edges(nodes, start_states, terminal_states, rng):
    """
    Edges of the union of an out-forest rooted at the start states and an in-forest rooted at the terminal states.

    Parameters
    ----------
    nodes : sequence of hashable
        All non-EOS states.
    start_states : sequence of hashable
        Roots of the out-forest.
    terminal_states : sequence of hashable
        Roots of the in-forest.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    list of tuple
        Distinct ``(source, target)`` pairs. The resulting graph is trim by construction.
    """
    out_hop = _sample_rooted_forest(nodes, start_states, rng)
    in_hop = _sample_rooted_forest(nodes, terminal_states, rng)
    edges = {(parent, v) for v, parent in out_hop.items()}
    edges |= {(v, parent) for v, parent in in_hop.items()}
    order = {v: i for i, v in enumerate(nodes)}
    return sorted(edges, key=lambda e: (order[e[0]], order[e[1]]))


def init_restricted_graph(
    states_no_eos,
    states_rep,
    alphabet,
    start_states,
    terminal_states,
    transition_density,
    rng,
    n_swap_steps=None,
):
    """
    Sample a trim random graph with ``ceil(n^2 * transition_density)`` edges and at most one edge per slot.

    A slot ``(u, a)`` is an outgoing edge from state ``u`` to a state emitting the observable symbol ``a``. Each slot
    holds at most one edge (self-loops allowed). The target distribution is that of :func:`init_tree_graph`
    conditioned on this slot rule: uniform over all trim graphs with exactly ``m = ceil(n^2 * transition_density)``
    edges (``n = len(states_no_eos)``) that have at most one edge per slot. The edge count is therefore the same as in
    the default sampler for the same `transition_density`.

    The sampler mirrors :func:`init_tree_graph`: (i) a uniform random spanning out-forest rooted at the start states
    and a uniform random spanning in-forest rooted at the terminal states are drawn, whose union is trim by
    construction; (ii) edges are removed in random order while the graph stays trim if there are more than ``m``, or
    uniformly random edges whose slot is still empty are added if there are fewer; (iii) an edge-swap Markov chain is
    run. A swap removes a uniformly chosen edge ``e``, then adds an edge ``f`` drawn uniformly among all edges that fit
    into the resulting graph (``f`` absent and its slot empty; ``f == e`` is possible and leaves the graph unchanged);
    the move is kept iff the graph is trim and reverted otherwise.

    Parameters
    ----------
    states_no_eos : list of hashable
        All states excluding EOS.
    states_rep : list of str
        Observable symbol emitted by each state of `states_no_eos` (same order).
    alphabet : list of str
        Observable symbols.
    start_states : list of hashable
        Start states (roots of the out-forest).
    terminal_states : list of hashable
        Terminal states (roots of the in-forest).
    transition_density : float
        Fraction of all ``n^2`` ordered state pairs that are edges (same meaning as in :func:`init_tree_graph`).
    rng : numpy.random.Generator
        Random number generator.
    n_swap_steps : int, optional
        Number of edge-swap proposals. Defaults to ``max(10 * m, 100)``.

    Returns
    -------
    G : nx.DiGraph or None
        Trim directed graph over `states_no_eos` with exactly ``m`` edges and at most one edge per slot, or None if two
        forest edges compete for the same slot with different targets, or if the forests union could not be pruned
        down to ``m`` edges within the iteration budget (retry in both cases).

    Raises
    ------
    ValueError
        If ``m`` is below the minimum number of edges of a trim graph (``max(n - |S|, n - |T|)``, plus one if
        ``S == T`` and ``S`` does not cover all states; the same floor as :func:`init_tree_graph`, verified by brute
        force also for restricted graphs, ``n <= 4``, all labellings and all non-empty ``S`` and ``T``) or exceeds the
        number of slots ``n * |alphabet|``, where every slot is filled.

    Notes
    -----
    The distribution is uniform over trim restricted graphs with ``m`` edges, i.e. the default sampler conditioned on
    the slot rule. A slot of a symbol emitted by ``k`` states can be filled in ``k`` ways, so such a symbol is the next
    symbol more often than a single-copy symbol, but less than ``k`` times as often, and each copy receives fewer
    in-edges than a single-state symbol, because the copies share one slot per source. The gap grows with density
    (e.g. for 6 states with two 2-copy symbols, the in-degree of a copy relative to a single-copy state is about 0.9,
    0.8 and 0.6 at mean out-degree 1.5, 2 and 3). A symbol-neutral variant is not implemented.

    The swap proposal passes through the same intermediate graph ``E - {e}`` in the forward and in the reverse
    direction, so it is symmetric and the stationary distribution is uniform over trim restricted graphs with exactly
    ``m`` edges (within the swap-connected component of the initial graph). The initial graph is not uniform and the
    chain is run for a finite number of steps, so samples are only approximately uniform; mixing is slow when ``m`` is
    close to the minimum, where trim graphs are necessarily chain-like.

    Greedy pruning of the forests union can get stuck at a minimal trim graph with more than ``m`` edges. In that case
    alternating swap proposals (as above) and single-edge removals are run for up to ``20 * n^2`` iterations; None is
    returned only if this budget is exhausted. Start states may have no incoming edges.
    """
    nodes = list(states_no_eos)
    n = len(nodes)
    m = int(np.ceil(n * n * transition_density))
    n_start = len(set(start_states))
    n_terminal = len(set(terminal_states))
    m_min = _min_trim_edges(n, start_states, terminal_states)
    symbols = sorted(set(states_rep) | set(alphabet))
    sym_idx = {a: i for i, a in enumerate(symbols)}
    n_symbols = len({sym_idx[a] for a in states_rep})  # symbols emitted by at least one state
    n_slots = n * n_symbols

    if m < m_min:
        raise ValueError(
            f"transition_density={transition_density} gives {m} edges for {n} states, but a trim grammar with "
            f"{n_start} start and {n_terminal} terminal states needs at least {m_min} (density too low)."
        )
    if m > n_slots:
        raise ValueError(
            f"Too many edges for a restricted grammar: {m} edges requested, but there are only {n_slots} slots "
            f"({n} states x {n_symbols} symbols)."
        )
    if n_swap_steps is None:
        n_swap_steps = max(10 * m, 100)

    idx = {v: i for i, v in enumerate(nodes)}
    sym_of = [sym_idx[a] for a in states_rep]
    start_mask = 0
    for s in set(start_states):
        start_mask |= 1 << idx[s]
    terminal_mask = 0
    for t in set(terminal_states):
        terminal_mask |= 1 << idx[t]

    edges = set()
    occ = [[-1] * len(symbols) for _ in range(n)]  # occ[u][a]: target of the edge in slot (u, a), -1 if empty
    for u, v in _forest_edges(nodes, start_states, terminal_states, rng):
        u, v = idx[u], idx[v]
        if occ[u][sym_of[v]] not in (-1, v):
            return None  # two different targets compete for the same slot; let the outer loop try again
        occ[u][sym_of[v]] = v
        edges.add((u, v))
    succ = [0] * n
    pred = [0] * n
    for u, v in edges:
        succ[u] |= 1 << v
        pred[v] |= 1 << u

    def remove(u, v):
        edges.discard((u, v))
        occ[u][sym_of[v]] = -1
        succ[u] &= ~(1 << v)
        pred[v] &= ~(1 << u)

    def add(u, v):
        edges.add((u, v))
        occ[u][sym_of[v]] = v
        succ[u] |= 1 << v
        pred[v] |= 1 << u

    n_pairs = n * n

    def draw_allowed():
        """Uniform edge whose slot is empty (rejection from the ``n^2`` ordered pairs)."""
        while True:
            p = int(rng.integers(0, n_pairs))
            u, v = p // n, p % n
            if occ[u][sym_of[v]] == -1:
                return u, v

    def swap(edge_list, k):
        """Remove ``edge_list[k]`` and add a uniform allowed edge; keep iff trim, else revert."""
        e = edge_list[k]
        remove(*e)
        f = draw_allowed()  # f == e is possible and is then a no-op
        if f == e:
            add(*e)
            return
        add(*f)
        if _is_trim(succ, pred, start_mask, terminal_mask):
            edge_list[k] = f
        else:
            remove(*f)
            add(*e)

    if len(edges) > m:
        # prune edges in random order, keeping the graph trim
        order = sorted(edges)
        for k in rng.permutation(len(order)):
            if len(edges) == m:
                break
            u, v = order[k]
            remove(u, v)
            if not _is_trim(succ, pred, start_mask, terminal_mask):
                add(u, v)

    if len(edges) > m:
        # Greedy pruning can get stuck at a minimal trim graph with more than m edges. Alternate swap proposals
        # (accepted iff trim) with single-edge removals to escape such local minima.
        edge_list = sorted(edges)
        for _ in range(20 * n * n):
            if len(edge_list) == m:
                break
            swap(edge_list, int(rng.integers(0, len(edge_list))))
            k = int(rng.integers(0, len(edge_list)))
            e = edge_list[k]
            remove(*e)
            if _is_trim(succ, pred, start_mask, terminal_mask):
                edge_list.pop(k)
            else:
                add(*e)
        if len(edges) > m:
            return None

    while len(edges) < m:
        allowed = [
            (u, v) for u in range(n) for v in range(n) if occ[u][sym_of[v]] == -1
        ]  # non-empty because m <= n_slots
        add(*allowed[int(rng.integers(0, len(allowed)))])

    edge_list = sorted(edges)
    for _ in range(n_swap_steps):
        swap(edge_list, int(rng.integers(0, m)))

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from((nodes[u], nodes[v]) for u, v in sorted(edges))
    return G


def init_tree_graph(states_no_eos, start_states, terminal_states, transition_density, rng, n_swap_steps=None):
    """
    Sample a random trim directed graph with ``ceil(n^2 * transition_density)`` edges.

    Trim means that every state is reachable from a start state and can reach a terminal state; strong connectivity
    is not required, and start states (terminal states) may have no incoming (outgoing) edges.

    The sampler (i) draws a uniform random spanning out-forest rooted at the start states and a uniform random
    spanning in-forest rooted at the terminal states, whose union is trim by construction; (ii) removes edges in
    random order as long as the graph stays trim until the target edge count is reached, or adds uniformly random
    absent edges if there are too few; (iii) runs an edge-swap Markov chain that preserves the edge count and trim
    property.

    Parameters
    ----------
    states_no_eos : list of hashable
        All states excluding the EOS (end-of-sequence) state.
    start_states : list of hashable
        Subset of states that can serve as start states.
    terminal_states : list of hashable
        Subset of states that can serve as terminal states. May overlap with `start_states`.
    transition_density : float
        Fraction of all ``n^2`` ordered state pairs (self-loops included) that are edges, with
        ``n = len(states_no_eos)``.
    rng : numpy.random.Generator
        Random number generator.
    n_swap_steps : int, optional
        Number of edge-swap proposals. Defaults to ``max(10 * m, 100)``, where ``m`` is the target number of edges.

    Returns
    -------
    G : nx.DiGraph or None
        Trim directed graph over `states_no_eos` with exactly ``m`` edges, or None if the forests union could not be
        pruned down to ``m`` edges (retry recommended).

    Raises
    ------
    ValueError
        If ``m`` is below the minimum number of edges of a trim graph (see Notes) or exceeds ``n^2``.

    Notes
    -----
    The minimum number of edges of a trim graph is ``max(n - |S|, n - |T|)`` for start set ``S`` and terminal set
    ``T``, plus one if ``S == T`` and ``S`` does not cover all states (the states outside ``S`` then need a chain that
    leaves ``S`` and returns to it). This was verified by brute-force enumeration of all edge sets for ``n <= 4`` and
    all non-empty choices of ``S`` and ``T`` (284 cases, no mismatch).

    Greedy pruning of the forests union can get stuck at a minimal trim graph with more than ``m`` edges. In that case
    alternating swap proposals and single-edge removals (each kept iff the graph stays trim) are run for up to
    ``20 * n^2`` iterations; None is returned only if this budget is exhausted.

    The swap proposal (a uniformly chosen edge replaced by a uniformly chosen absent pair) is symmetric, and a proposal
    is accepted iff the result is trim, so the chain's stationary distribution is uniform over trim graphs with
    exactly ``m`` edges (within the swap-connected component of the initial graph). The initial graph is not uniform
    and the chain is run for a finite number of steps, so samples are only approximately uniform; mixing is slow when
    ``m`` is close to the minimum, where trim graphs are necessarily chain-like.
    """
    nodes = list(states_no_eos)
    n = len(nodes)
    m = int(np.ceil(n * n * transition_density))
    n_start = len(set(start_states))
    n_terminal = len(set(terminal_states))
    m_min = _min_trim_edges(n, start_states, terminal_states)  # exact (verified by brute force for n <= 4)

    if m < m_min:
        raise ValueError(
            f"transition_density={transition_density} gives {m} edges for {n} states, but a trim grammar with "
            f"{n_start} start and {n_terminal} terminal states needs at least {m_min} (density too low)."
        )
    if m > n * n:
        raise ValueError(f"Too many edges: maximum is n^2 = {n * n}.")
    if n_swap_steps is None:
        n_swap_steps = max(10 * m, 100)

    idx = {v: i for i, v in enumerate(nodes)}
    start_mask = 0
    for s in set(start_states):
        start_mask |= 1 << idx[s]
    terminal_mask = 0
    for t in set(terminal_states):
        terminal_mask |= 1 << idx[t]

    edges = {(idx[u], idx[v]) for u, v in _forest_edges(nodes, start_states, terminal_states, rng)}
    succ = [0] * n
    pred = [0] * n
    for u, v in edges:
        succ[u] |= 1 << v
        pred[v] |= 1 << u

    def remove(u, v):
        edges.discard((u, v))
        succ[u] &= ~(1 << v)
        pred[v] &= ~(1 << u)

    def add(u, v):
        edges.add((u, v))
        succ[u] |= 1 << v
        pred[v] |= 1 << u

    if len(edges) > m:
        # prune edges in random order, keeping the graph trim
        order = sorted(edges)
        for k in rng.permutation(len(order)):
            if len(edges) == m:
                break
            u, v = order[k]
            remove(u, v)
            if not _is_trim(succ, pred, start_mask, terminal_mask):
                add(u, v)

    if len(edges) > m:
        # Greedy pruning can get stuck at a minimal trim graph with more than m edges. Alternate swap proposals
        # (accepted iff trim) with single-edge removals to escape such local minima.
        edge_list = sorted(edges)
        n_pairs = n * n
        for _ in range(20 * n * n):
            if len(edge_list) == m:
                break
            if len(edge_list) < n_pairs:
                k = int(rng.integers(0, len(edge_list)))
                e = edge_list[k]
                while True:  # uniform absent pair by rejection
                    p = int(rng.integers(0, n_pairs))
                    f = (p // n, p % n)
                    if f not in edges:
                        break
                remove(*e)
                add(*f)
                if _is_trim(succ, pred, start_mask, terminal_mask):
                    edge_list[k] = f
                else:
                    remove(*f)
                    add(*e)
            k = int(rng.integers(0, len(edge_list)))
            e = edge_list[k]
            remove(*e)
            if _is_trim(succ, pred, start_mask, terminal_mask):
                edge_list.pop(k)
            else:
                add(*e)
        if len(edges) > m:
            return None

    if len(edges) < m:
        absent = [(u, v) for u in range(n) for v in range(n) if (u, v) not in edges]
        for k in rng.choice(len(absent), size=m - len(edges), replace=False):
            add(*absent[k])

    if len(edges) < n * n:
        edge_list = sorted(edges)
        n_pairs = n * n
        for _ in range(n_swap_steps):
            k = int(rng.integers(0, m))
            e = edge_list[k]
            while True:  # uniform absent pair by rejection
                p = int(rng.integers(0, n_pairs))
                f = (p // n, p % n)
                if f not in edges:
                    break
            remove(*e)
            add(*f)
            if _is_trim(succ, pred, start_mask, terminal_mask):
                edge_list[k] = f
            else:
                remove(*f)
                add(*e)

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from((nodes[u], nodes[v]) for u, v in sorted(edges))
    return G


def init_cycle_graph(states_no_eos, start_states, terminal_states, transition_density, rng):
    """
    Generate a random sparse but strongly connected directed graph (legacy sampler).

    A random Hamiltonian ring through all states is planted (a start state first, a terminal state last), then
    uniformly random extra edges (self-loops included) are added up to ``ceil(n^2 * transition_density)`` edges.
    This sampler is kept for reproducibility of earlier results; prefer :func:`init_tree_graph`.

    Parameters
    ----------
    states_no_eos : list of hashable
        List of all states excluding the EOS (end-of-sequence) state.
    start_states : list of hashable
        Subset of states that can serve as valid starting states.
    terminal_states : list of hashable
        Subset of states that can serve as valid terminal states.
    transition_density : float
        Fraction of all ``n^2`` ordered state pairs to include as edges, where ``n = len(states_no_eos)``.
        Must satisfy ``n / n^2 <= transition_density <= (n-1) / n``.
    rng : numpy.random.Generator
        Random number generator instance used for reproducibility.

    Returns
    -------
    G : nx.DiGraph or None
        A strongly connected directed graph with `len(states_no_eos)` nodes
        and ``ceil(n^2 * transition_density)`` edges.
        Returns `None` if the randomly chosen start state coincides with
        the chosen terminal state (retry recommended).

    Raises
    ------
    RuntimeError
        If the requested number of edges is less than `n`, making strong
        connectivity impossible.
    ValueError
        If the requested number of edges exceeds `n * (n - 1)`.

    Notes
    -----
    Because of the planted ring, every grammar produced this way has the following properties, which limit the
    diversity of the sampled grammars:

    - every state, including the start states, has at least one incoming edge (the grammar is strongly connected);
    - the terminal-to-start ring edge makes the language closed under concatenation (``L = L+``) when there is a
      single start and a single terminal state;
    - grammars with tree-like or otherwise not strongly connected structure (such as the preset grammars) can never
      be produced.
    """
    n = len(states_no_eos)
    m = int(np.ceil(n * n * transition_density))

    if m < n:
        # raise ValueError("Need at least n edges for strong connectivity.")
        raise RuntimeError(
            "Could not generate a valid transition matrix (strongly connected graph)"
            "with the provided density (too low?)."
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

    # build all possible edges (self-loops included)
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
    Thin wrapper for `symseq.generators.ag.synthesis.generate_grammar` to generate a grammar with a target TE complexity.
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
    A = _normalize_rows_l1(adj_mat)

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
