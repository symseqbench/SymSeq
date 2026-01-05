# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Grammar complexity measures."""

import warnings


def grammar_rule_complexity(grammar: dict) -> dict:
    """
    Compute rule-based complexity measures for a grammar.

    Parameters
    ----------
    grammar : dict
        Grammar representation.
        Format: {'S': [['A', 'B'], ['a']], 'A': [['a']], ...}

    Returns
    -------
    dict
        Complexity measures:
        - 'num_rules': Number of production rules
        - 'num_nonterminals': Number of non-terminal symbols
        - 'num_terminals': Number of terminal symbols
        - 'avg_rule_length': Average right-hand side length
        - 'branching_factor': Average number of rules per non-terminal

    Notes
    -----
    Basic structural complexity measures.

    Examples
    --------
    >>> grammar = {'S': [['A', 'B'], ['a']], 'A': [['a']], 'B': [['b']]}
    >>> complexity = grammar_rule_complexity(grammar)
    """
    if not grammar:
        return {
            'num_rules': 0,
            'num_nonterminals': 0,
            'num_terminals': 0,
            'avg_rule_length': 0.0,
            'branching_factor': 0.0,
        }

    num_nonterminals = len(grammar)
    num_rules = sum(len(rules) for rules in grammar.values())

    terminals = set()
    rule_lengths = []

    for nonterminal, rules in grammar.items():
        for rule in rules:
            rule_lengths.append(len(rule))
            for symbol in rule:
                if symbol not in grammar:
                    terminals.add(symbol)

    num_terminals = len(terminals)
    avg_rule_length = sum(rule_lengths) / len(rule_lengths) if rule_lengths else 0.0
    branching_factor = num_rules / num_nonterminals if num_nonterminals > 0 else 0.0

    return {
        'num_rules': num_rules,
        'num_nonterminals': num_nonterminals,
        'num_terminals': num_terminals,
        'avg_rule_length': avg_rule_length,
        'branching_factor': branching_factor,
    }


def grammar_state_complexity(grammar: dict) -> int:
    """
    Estimate minimal automaton size for grammar.

    Parameters
    ----------
    grammar : dict
        Grammar representation.

    Returns
    -------
    int
        Estimated state complexity.

    Notes
    -----
    This is a placeholder. Full implementation requires:
    - Converting CFG to PDA
    - Minimizing automaton
    - Computing state complexity

    For regular grammars, this is the minimal DFA size.
    For context-free grammars, this is the minimal PDA size.
    """
    warnings.warn(
        "State complexity estimation is not fully implemented.",
        UserWarning
    )

    return len(grammar)


def derivation_tree_complexity(grammar: dict, n_samples: int = 100) -> dict:
    """
    Analyze derivation tree complexity via sampling.

    Parameters
    ----------
    grammar : dict
        Grammar representation.
    n_samples : int, default=100
        Number of derivations to sample.

    Returns
    -------
    dict
        Tree complexity measures:
        - 'avg_depth': Average derivation tree depth
        - 'avg_width': Average tree width
        - 'avg_nodes': Average number of nodes

    Notes
    -----
    This is a placeholder. Full implementation requires:
    - Random derivation sampling
    - Tree structure analysis
    - Statistical aggregation
    """
    warnings.warn(
        "Derivation tree complexity is not fully implemented.",
        UserWarning
    )
    raise NotImplementedError(
        "Derivation tree analysis requires grammar derivation sampling."
    )
