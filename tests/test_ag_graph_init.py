"""Tests for the graph initialisers of random artificial grammars (tree, restricted and cycle)."""

import itertools
import math
import warnings
from collections import Counter

import networkx as nx
import numpy as np
import pytest
from scipy import stats

import symseq.generators.ag.artificial_grammar as artificial_grammar_module
from symseq.config import load_trial_set
from symseq.generators.ag import ArtificialGrammar
from symseq.generators.ag.generator import (
    _has_equivalent_copies,
    _sample_rooted_forest,
    generate_random_grammar,
    init_restricted_graph,
    init_tree_graph,
)

EOS = "#"

# alphabet size, ambiguities, ambiguity depth, density, n start states, n terminal states
CONFIGS = [
    (4, 0, 0, 0.35, 1, 1),
    (4, 0, 0, 0.5, 2, 2),
    (4, 2, 2, 0.25, 1, 1),
    (4, 2, 2, 0.35, 2, 1),
    (4, 2, 2, 0.5, 1, 2),
    (6, 0, 0, 0.25, 1, 1),
    (6, 0, 0, 0.35, 2, 2),
    (6, 2, 2, 0.5, 1, 1),
    (6, 2, 2, 0.25, 2, 2),
]
N_SEEDS = 12

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


# --------------------- helpers ---------------------


def _symbol(state_name):
    """Observable symbol of a state name such as ``"B(1)"``."""
    return state_name.split("(")[0]


def _graph(params):
    """Directed graph (all states, EOS included) of a grammar parameter dictionary."""
    G = nx.DiGraph()
    G.add_nodes_from(params["states"])
    G.add_edges_from((u, v) for u, v, _ in params["transitions"])
    return G


def _non_eos_graph(params):
    return _graph(params).subgraph([s for s in params["states"] if s != EOS]).copy()


def _random_params(config, seed, mode="tree", restricted=False):
    alphabet, amb, depth, density, n_start, n_term = config
    return generate_random_grammar(
        alphabet_size=alphabet,
        ambiguities=amb,
        ambiguity_depth=depth,
        transition_density=density,
        n_start_states=n_start,
        n_terminal_states=n_term,
        seed=seed,
        verbose=False,
        graph_init=mode,
        restricted=restricted,
        max_iter=20000,
    )


def _assert_trim(params):
    """Assert the trim property with explicit sink / closed-component checks, independent of the generator."""
    G = _graph(params)
    starts = list(params["start_states"])
    # every state is reachable from a start state
    reachable = set(starts)
    for s in starts:
        reachable |= nx.descendants(G, s)
    assert reachable == set(G.nodes)
    # every state reaches EOS
    assert nx.ancestors(G, EOS) | {EOS} == set(G.nodes)
    # no non-terminal sinks
    assert all(G.out_degree(v) > 0 for v in G.nodes if v != EOS)
    # no closed (bottom) strongly connected component that cannot reach EOS
    C = nx.condensation(G)
    eos_comp = C.graph["mapping"][EOS]
    for comp in C.nodes:
        if C.out_degree(comp) == 0:
            assert comp == eos_comp, f"closed component without EOS: {C.nodes[comp]['members']}"


def _is_trim_edges(edges, n, starts, terminals):
    """Independent trim check on an edge set over nodes ``0..n-1``."""
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    fwd = set(starts)
    for s in starts:
        fwd |= nx.descendants(G, s)
    bwd = set(terminals)
    for t in terminals:
        bwd |= nx.ancestors(G, t)
    return len(fwd) == n and len(bwd) == n


def _sample_tree(nodes, starts, terminals, density, rng, **kwargs):
    """Draw from init_tree_graph, retrying on None."""
    for _ in range(300):
        G = init_tree_graph(nodes, starts, terminals, density, rng, **kwargs)
        if G is not None:
            return G
    raise AssertionError("init_tree_graph returned None 300 times in a row")


def _assert_trim_graph(G, nodes, starts, terminals, m):
    assert set(G.nodes) == set(nodes)
    assert G.number_of_edges() == m
    idx = {v: i for i, v in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for u, v in G.edges]
    assert _is_trim_edges(edges, len(nodes), [idx[s] for s in starts], [idx[t] for t in terminals])


def _m_min(n, starts, terminals):
    s, t = set(starts), set(terminals)
    return max(n - len(s), n - len(t)) + (1 if s == t and n > len(s) else 0)


# --------------------- 1. trim property ---------------------


@pytest.mark.parametrize(
    "mode,restricted", [("tree", False), ("restricted", True), ("cycle", False)], ids=["tree", "restricted", "cycle"]
)
@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: "a{}_amb{}x{}_d{}_s{}_t{}".format(*c))
def test_trim_property(config, mode, restricted):
    for seed in range(N_SEEDS):
        params = _random_params(config, seed, mode="cycle" if mode == "cycle" else "tree", restricted=restricted)
        _assert_trim(params)


# --------------------- 2. edge counts and restricted slots ---------------------


@pytest.mark.parametrize("restricted", [False, True], ids=["tree", "restricted"])
@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: "a{}_amb{}x{}_d{}_s{}_t{}".format(*c))
def test_exact_edge_count(config, restricted):
    """Both modes realise exactly ceil(n^2 d) non-EOS edges (same d -> same edge count)."""
    alphabet, amb, depth, density, _, _ = config
    n = alphabet - amb + amb * depth
    expected = math.ceil(n * n * density)
    for seed in range(N_SEEDS):
        params = _random_params(config, seed, restricted=restricted)
        n_edges = sum(1 for _, v, _ in params["transitions"] if v != EOS)
        assert n_edges == expected


@pytest.mark.parametrize(
    "config", [c for c in CONFIGS if c[1] > 0] + CONFIGS[:2], ids=lambda c: "a{}_amb{}_d{}".format(c[0], c[1], c[3])
)
def test_restricted_mode_one_target_per_slot(config):
    for seed in range(N_SEEDS):
        params = _random_params(config, seed, restricted=True)
        slots = Counter((u, _symbol(v)) for u, v, _ in params["transitions"] if v != EOS)
        assert max(slots.values()) == 1


# --------------------- 3. ring removed in tree mode, present in cycle mode ---------------------


def _concatenation_accepted(grammar):
    w1 = grammar.generate_string()
    w2 = grammar.generate_string()
    return grammar.is_grammatical(list(w1) + list(w2))


def _structure_stats(mode, n_seeds=60):
    """Return (start_in_degree_0, not_strongly_connected, concatenation_rejected, all_have_term_to_start)."""
    start_indeg0 = not_strong = concat_rejected = False
    term_to_start = True
    for seed in range(n_seeds):
        params = generate_random_grammar(
            alphabet_size=6, transition_density=0.3, seed=seed, verbose=False, graph_init=mode, max_iter=20000
        )
        G = _non_eos_graph(params)
        s, t = params["start_states"][0], params["terminal_states"][0]
        start_indeg0 |= G.in_degree(s) == 0
        not_strong |= not nx.is_strongly_connected(G)
        term_to_start &= G.has_edge(t, s)
        grammar = ArtificialGrammar(rng=None, seed=seed, verbose=False, **params)
        concat_rejected |= not _concatenation_accepted(grammar)
    return start_indeg0, not_strong, concat_rejected, term_to_start


def test_tree_mode_has_no_ring():
    start_indeg0, not_strong, concat_rejected, term_to_start = _structure_stats("tree")
    assert start_indeg0, "no grammar with a start state of in-degree 0"
    assert not_strong, "every grammar was strongly connected"
    assert concat_rejected, "language was closed under concatenation in every grammar"
    assert not term_to_start


def test_cycle_mode_has_ring():
    start_indeg0, not_strong, concat_rejected, term_to_start = _structure_stats("cycle")
    assert not start_indeg0
    assert not not_strong
    assert not concat_rejected
    assert term_to_start


# --------------------- 4. direct calls of the initialisers ---------------------

NODES5 = ["a", "b", "c", "d", "e"]


@pytest.mark.parametrize(
    "starts,terminals",
    [
        (["a"], ["e"]),
        (["a", "b"], ["d", "e"]),  # multiple starts and terminals
        (["a", "b", "c"], ["e"]),
        (["a"], ["c", "d", "e"]),
        (["b", "c"], ["c", "d"]),  # overlapping sets
        (["c"], ["c"]),  # start == terminal
        (["a", "b"], ["a", "b"]),  # S == T, several states
    ],
    ids=lambda x: "".join(x),
)
@pytest.mark.parametrize("density", [0.3, 0.5, 0.8])
def test_direct_tree_and_restricted_inits_are_trim(starts, terminals, density):
    rng = np.random.default_rng(5)
    m = math.ceil(25 * density)
    for _ in range(10):
        _assert_trim_graph(_sample_tree(NODES5, starts, terminals, density, rng), NODES5, starts, terminals, m)

    rep = ["A", "B", "B", "C", "D"]  # states b and c share the symbol B
    got = 0
    idx = {v: i for i, v in enumerate(NODES5)}
    for _ in range(100):
        G = init_restricted_graph(NODES5, rep, ["A", "B", "C", "D"], starts, terminals, density, rng)
        if G is None:
            continue
        got += 1
        _assert_trim_graph(G, NODES5, starts, terminals, m)
        slots = Counter((u, rep[idx[v]]) for u, v in G.edges)
        assert max(slots.values()) == 1
    assert got > 0


@pytest.mark.parametrize(
    "starts,terminals",
    [
        (["a"], ["e"]),  # S != T, single
        (["a", "b"], ["d", "e"]),  # S != T, multiple
        (["b", "c"], ["c", "d"]),  # overlapping, S != T
        (["c"], ["c"]),  # S == T, single
        (["a", "b"], ["a", "b"]),  # S == T, multiple
    ],
    ids=lambda x: "".join(x),
)
def test_tree_graph_at_minimum_edge_count(starts, terminals):
    n = len(NODES5)
    m = _m_min(n, starts, terminals)
    density = (m - 0.5) / (n * n)  # ceil(n^2 d) == m
    rng = np.random.default_rng(11)
    for _ in range(10):
        _assert_trim_graph(_sample_tree(NODES5, starts, terminals, density, rng), NODES5, starts, terminals, m)


@pytest.mark.parametrize(
    "starts,terminals",
    [(["a"], ["e"]), (["b", "c"], ["c", "d"]), (["c"], ["c"]), (["a", "b"], ["a", "b"]), (["a"], ["b", "c"])],
    ids=lambda x: "".join(x),
)
def test_restricted_graph_at_minimum_and_maximum_edge_count(starts, terminals):
    rep = ["A", "B", "B", "C", "C"]  # symbols with 1, 2 and 2 copies; 5 states x 3 symbols = 15 slots
    alphabet = ["A", "B", "C"]
    n = len(NODES5)
    rng = np.random.default_rng(3)
    for m in (_m_min(n, starts, terminals), 15):
        density = (m - 0.5) / (n * n)  # ceil(n^2 d) == m
        got = 0
        for _ in range(60):
            G = init_restricted_graph(NODES5, rep, alphabet, starts, terminals, density, rng)
            if G is None:
                continue
            got += 1
            _assert_trim_graph(G, NODES5, starts, terminals, m)
        assert got > 0


def test_restricted_graph_raises_outside_edge_range():
    rep = ["A", "B", "B", "C", "C"]
    alphabet = ["A", "B", "C"]
    rng = np.random.default_rng(0)
    n, n_slots = 5, 15
    with pytest.raises(ValueError, match="slots"):  # m = n_slots + 1
        init_restricted_graph(NODES5, rep, alphabet, ["a"], ["e"], (n_slots + 0.5) / (n * n), rng)
    with pytest.raises(ValueError, match="density too low"):  # m = 3 < 4
        init_restricted_graph(NODES5, rep, alphabet, ["a"], ["e"], 0.1, rng)
    with pytest.raises(ValueError, match="density too low"):  # S == T, n > |S|: floor is 5
        init_restricted_graph(NODES5, rep, alphabet, ["c"], ["c"], (4 - 0.5) / (n * n), rng)
    with pytest.raises(ValueError, match="slots"):  # n = 6 states, 4 symbols: 24 slots < ceil(36 * 0.8) = 29 edges
        generate_random_grammar(
            alphabet_size=4, ambiguities=2, ambiguity_depth=2, transition_density=0.8, verbose=False, restricted=True
        )
    generate_random_grammar(alphabet_size=4, transition_density=1.0, verbose=False, restricted=True)  # all 16 slots


def test_tree_graph_equal_start_terminal_needs_extra_edge():
    """With S == T and n > |S| a trim graph needs one edge more than n - |S|."""
    rng = np.random.default_rng(0)
    n, m = 5, 4
    with pytest.raises(ValueError):
        init_tree_graph(NODES5, ["c"], ["c"], (m - 0.5) / (n * n), rng)


# --------------------- 5. determinism ---------------------


@pytest.mark.parametrize("mode,restricted", [("tree", False), ("tree", True), ("cycle", False)])
def test_same_seed_same_grammar(mode, restricted):
    kwargs = dict(
        alphabet_size=6,
        ambiguities=2,
        ambiguity_depth=2,
        transition_density=0.3,
        seed=123,
        verbose=False,
        graph_init=mode,
        restricted=restricted,
    )
    p1, p2 = generate_random_grammar(**kwargs), generate_random_grammar(**kwargs)
    assert p1 == p2
    p3 = generate_random_grammar(**{**kwargs, "seed": 124})
    assert p3["transitions"] != p1["transitions"] or p3["start_states"] != p1["start_states"]


def test_init_tree_graph_deterministic_given_rng():
    edges = [
        sorted(init_tree_graph(NODES5, ["a"], ["e"], 0.4, np.random.default_rng(9)).edges) for _ in range(2)
    ]
    assert edges[0] == edges[1]


# --------------------- 6. errors, warnings, forwarding ---------------------


def test_unknown_graph_init_raises():
    with pytest.raises(ValueError, match="graph_init"):
        generate_random_grammar(graph_init="spiral", verbose=False)


def test_restricted_with_cycle_raises():
    with pytest.raises(ValueError, match="restricted"):
        generate_random_grammar(restricted=True, graph_init="cycle", verbose=False)


def test_restricted_with_target_complexity_raises():
    with pytest.raises(ValueError, match="target_complexity"):
        ArtificialGrammar.from_constraints(restricted=True, target_complexity=1.0, verbose=False)


def test_unknown_graph_init_raises_from_constraints():
    with pytest.raises(ValueError, match="graph_init"):
        ArtificialGrammar.from_constraints(graph_init="spiral", verbose=False)


def test_density_below_minimum_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="density too low"):
        init_tree_graph(NODES5, ["a"], ["e"], 0.1, rng)  # m = 3 < 4
    with pytest.raises(ValueError, match="density too low"):
        generate_random_grammar(alphabet_size=6, transition_density=0.05, verbose=False)
    with pytest.raises(ValueError):
        init_tree_graph(NODES5, ["a"], ["e"], 1.2, rng)  # m > n^2


@pytest.mark.parametrize("restricted", [False, True], ids=["tree", "restricted"])
def test_low_density_warns(restricted):
    # n = 6, m = ceil(36 * 0.2) = 8 < 1.5 * 6
    with pytest.warns(UserWarning, match="chain-like"):
        generate_random_grammar(alphabet_size=6, transition_density=0.2, verbose=False, restricted=restricted)


def test_no_low_density_warning_otherwise():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        generate_random_grammar(alphabet_size=6, transition_density=0.35, verbose=False)  # m = 13 >= 9
        generate_random_grammar(alphabet_size=6, transition_density=0.35, verbose=False, restricted=True)
        generate_random_grammar(alphabet_size=6, transition_density=0.2, verbose=False, graph_init="cycle")


def _capture_generate_random_grammar(monkeypatch):
    calls = []
    real = artificial_grammar_module.generate_random_grammar

    def spy(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(artificial_grammar_module, "generate_random_grammar", spy)
    return calls


def test_graph_init_forwarded_by_from_constraints(monkeypatch):
    calls = _capture_generate_random_grammar(monkeypatch)
    ArtificialGrammar.from_constraints(alphabet_size=5, transition_density=0.4, graph_init="cycle", verbose=False)
    ArtificialGrammar.from_constraints(alphabet_size=5, transition_density=0.4, verbose=False)
    ArtificialGrammar.from_constraints(
        alphabet_size=5, transition_density=0.4, restricted=True, graph_init="tree", verbose=False
    )
    assert [c["graph_init"] for c in calls] == ["cycle", "tree", "tree"]
    assert [c["restricted"] for c in calls] == [False, False, True]


def _random_cfg(**params):
    base = {
        "label": "Random AG test",
        "alphabet_size": 5,
        "transition_density": 0.4,
        "min_string_length": 2,
        "verbose": False,
    }
    base.update(params)
    return {
        "symseq": {
            "seed": 42,
            "generator": {"type": "ArtificialGrammar", "mode": "random", "params": base},
            "trial_constraints": {"length": {"min": 2, "max": 12}},
            "trial_set": {"n_trials": 3},
        }
    }


def test_graph_init_forwarded_by_config(monkeypatch):
    calls = _capture_generate_random_grammar(monkeypatch)
    ts = load_trial_set(_random_cfg(graph_init="cycle"))
    assert calls[-1]["graph_init"] == "cycle"
    source = ts.meta["generator"].source
    assert nx.is_strongly_connected(source.graph.subgraph([s for s in source.graph.nodes if s != EOS]))
    load_trial_set(_random_cfg())
    assert calls[-1]["graph_init"] == "tree"


def test_unknown_graph_init_raises_from_config():
    with pytest.raises(ValueError, match="graph_init"):
        load_trial_set(_random_cfg(graph_init="spiral"))


# --------------------- 7. uniformity of the edge-swap chain ---------------------


def _all_trim_edge_sets(n, starts, terminals, m):
    pairs = [(u, v) for u in range(n) for v in range(n)]
    return {
        frozenset(c)
        for c in itertools.combinations(pairs, m)
        if _is_trim_edges(c, n, starts, terminals)
    }


@pytest.mark.parametrize("m,n_samples", [(4, 4000), (3, 3000)])
def test_edge_swap_chain_is_uniform_on_tiny_case(m, n_samples):
    nodes = ["s0", "s1", "s2"]
    starts, terminals = ["s0"], ["s2"]
    density = (m - 0.5) / 9  # ceil(9 d) == m
    support = _all_trim_edge_sets(3, [0], [2], m)
    assert len(support) > 5
    idx = {v: i for i, v in enumerate(nodes)}

    rng = np.random.default_rng(2024)
    counts = Counter()
    for _ in range(n_samples):
        G = _sample_tree(nodes, starts, terminals, density, rng)
        counts[frozenset((idx[u], idx[v]) for u, v in G.edges)] += 1

    assert set(counts) == support
    observed = np.array([counts[e] for e in support])
    chi2, p = stats.chisquare(observed)  # uniform expectation
    assert p > 1e-3, f"chi2={chi2:.1f}, p={p:.2e}, {len(support)} cells"


def _restricted_edge_sets(n, labels, starts, terminals, m):
    """All trim edge sets with m edges and at most one edge per slot (source, symbol of target)."""
    pairs = [(u, v) for u in range(n) for v in range(n)]
    return {
        frozenset(c)
        for c in itertools.combinations(pairs, m)
        if len({(u, labels[v]) for u, v in c}) == m and _is_trim_edges(c, n, starts, terminals)
    }


def _restricted_chi2_pvalue(n_swap_steps, start="A0", terminal="B", n_support=22, n_samples=4400):
    """Chi-square p-value of init_restricted_graph against uniform on a tiny case; also checks the support."""
    nodes, rep = ["A0", "A1", "B"], ["A", "A", "B"]  # A has two copies
    m = 4
    support = _restricted_edge_sets(3, [0, 0, 1], [nodes.index(start)], [nodes.index(terminal)], m)
    assert len(support) == n_support
    rng = np.random.default_rng(2024)
    counts = Counter()
    while sum(counts.values()) < n_samples:
        G = init_restricted_graph(
            nodes, rep, ["A", "B"], [start], [terminal], (m - 0.5) / 9, rng, n_swap_steps=n_swap_steps
        )  # ceil(9 d) == m
        if G is not None:
            counts[frozenset((nodes.index(u), nodes.index(v)) for u, v in G.edges)] += 1
    assert set(counts) == support
    return stats.chisquare([counts[e] for e in support]).pvalue


@pytest.mark.parametrize("start,terminal,n_support", [("A0", "B", 22), ("B", "A0", 15)])
def test_restricted_sampler_is_uniform_on_tiny_case(start, terminal, n_support):
    """Uniform over trim restricted edge sets, i.e. the default sampler conditioned on the slot rule."""
    p = _restricted_chi2_pvalue(None, start, terminal, n_support)
    assert p > 1e-3, f"p={p:.2e}"


def test_restricted_uniformity_test_has_power():
    """Without the swap chain the initial graphs are far from uniform, which the test above must detect."""
    p = _restricted_chi2_pvalue(n_swap_steps=0)
    assert p < 1e-3, f"p={p:.2e}"


# --------------------- 8. rejection of bisimilar copies ---------------------


def _hand_built(edges):
    states = ["B", "A(0)", "A(1)", "C", EOS]
    return states, [(u, v, 1.0) for u, v in edges]


def test_has_equivalent_copies_detects_bisimilar_copies():
    # A(0) and A(1) both go to C, and C leads on to A(1) and EOS: the copies are interchangeable
    states, transitions = _hand_built([("B", "A(0)"), ("A(0)", "C"), ("A(1)", "C"), ("C", "A(1)"), ("C", EOS)])
    assert _has_equivalent_copies(states, transitions, EOS)


def test_has_equivalent_copies_accepts_genuine_ambiguity():
    # the copies differ in their futures: A(0) goes on to C, A(1) ends the string
    states, transitions = _hand_built([("B", "A(0)"), ("A(0)", "C"), ("C", "A(1)"), ("A(1)", EOS)])
    assert not _has_equivalent_copies(states, transitions, EOS)


def test_reject_equivalent_copies_removes_bisimilar_copies():
    for seed in range(30):
        params = generate_random_grammar(
            alphabet_size=4,
            ambiguities=2,
            ambiguity_depth=2,
            transition_density=0.5,
            seed=seed,
            verbose=False,
            restricted=True,
            reject_equivalent_copies=True,
        )
        assert not _has_equivalent_copies(params["states"], params["transitions"], EOS)


def test_reject_equivalent_copies_off_is_identical():
    kwargs = dict(alphabet_size=4, ambiguities=2, ambiguity_depth=2, transition_density=0.4, verbose=False)
    for restricted in (False, True):
        for seed in range(10):
            default = generate_random_grammar(seed=seed, restricted=restricted, **kwargs)
            off = generate_random_grammar(seed=seed, restricted=restricted, reject_equivalent_copies=False, **kwargs)
            assert default == off


def test_reject_equivalent_copies_with_target_complexity_raises():
    with pytest.raises(ValueError, match="target_complexity"):
        ArtificialGrammar.from_constraints(reject_equivalent_copies=True, target_complexity=1.0, verbose=False)


def test_reject_equivalent_copies_forwarded_by_from_constraints(monkeypatch):
    calls = _capture_generate_random_grammar(monkeypatch)
    ArtificialGrammar.from_constraints(alphabet_size=5, transition_density=0.4, verbose=False)
    ArtificialGrammar.from_constraints(
        alphabet_size=5, transition_density=0.4, reject_equivalent_copies=True, verbose=False
    )
    assert [c["reject_equivalent_copies"] for c in calls] == [False, True]


# --------------------- 9. Wilson's algorithm ---------------------


def _all_rooted_forests(nodes, roots):
    """Brute-force enumeration of next-hop maps in which every non-root node reaches a root (acyclic)."""
    non_roots = [v for v in nodes if v not in roots]
    forests = set()
    for choice in itertools.product(*[[u for u in nodes if u != v] for v in non_roots]):
        hop = dict(zip(non_roots, choice))
        ok = True
        for v in non_roots:
            seen, x = {v}, v
            while x not in roots:
                x = hop[x]
                if x in seen:
                    ok = False
                    break
                seen.add(x)
            if not ok:
                break
        if ok:
            forests.add(tuple(sorted(hop.items())))
    return forests


@pytest.mark.parametrize("roots,n_samples", [(["r"], 3000), (["r", "x"], 2400)])
def test_wilson_forest_is_uniform(roots, n_samples):
    nodes = ["r", "x", "y", "z"]
    support = _all_rooted_forests(nodes, set(roots))
    if len(roots) == 1:
        assert len(support) == 16  # n^(n-2) spanning arborescences
    else:
        assert len(support) == 8  # k * n^(n-k-1)

    rng = np.random.default_rng(77)
    counts = Counter(tuple(sorted(_sample_rooted_forest(nodes, roots, rng).items())) for _ in range(n_samples))
    assert set(counts) == support
    chi2, p = stats.chisquare([counts[f] for f in support])
    assert p > 1e-3, f"chi2={chi2:.1f}, p={p:.2e}"
