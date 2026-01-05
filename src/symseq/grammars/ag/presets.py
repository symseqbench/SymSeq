# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""
AGL_preset.py

These preset AGs labeled A-F correspond to the grammars in [1]

    [1] Schiff, R., & Katan, P. (2(014). Does complexity matter? Meta-analysis of learner performance in artificial
        grammar	tasks. Frontiers in Psychology, 5. https://doi.org/10.3389/fpsyg.2014.01084

Authors: Renato Duarte, Barna Zajzon
"""

Elman = {
    "label": "Elman",
    "alphabet": ["b", "a", "d", "i", "g", "u"],
    "states": ["b", "a", "d", "i(1)", "i(2)", "g", "u(1)", "u(2)", "u(3)"],
    "start_states": ["b", "d", "g"],
    "terminal_states": ["a", "i(2)", "u(3)"],
    "transitions": [
        ("b", "a", 1.0),
        ("d", "i(1)", 1.0),
        ("i(1)", "i(2)", 1.0),
        ("g", "u(1)", 1.0),
        ("u(1)", "u(2)", 1.0),
        ("u(2)", "u(3)", 1.0),
    ],
}

FSG_G1_A = {
    "label": "[A] Reber (1969)",
    "states": ["M(1)", "V(2)", "T(1)", "V(1)", "T(2)", "R(1)", "X(1)", "X(2)", "R(2)", "M(2)", "#"],
    "alphabet": ["M", "V", "T", "R", "X"],  # '#' removed from alphabet
    "start_states": ["M(1)", "V(2)"],
    "terminal_states": [
        "V(1)",
        "T(2)",
        "X(2)",
        "R(2)",
        "M(2)",
    ],
    "transitions": [
        ("M(1)", "T(1)", 0.5),
        ("M(1)", "V(1)", 0.5),
        ("T(1)", "T(1)", 0.5),
        ("T(1)", "V(1)", 0.5),
        ("V(1)", "T(2)", 1.0 / 3.0),
        ("V(1)", "R(1)", 1.0 / 3.0),
        ("V(1)", "#", 1.0 / 3.0),
        ("T(2)", "#", 1.0),
        ("R(1)", "X(1)", 0.5),
        ("R(1)", "X(2)", 0.5),
        ("V(2)", "X(1)", 0.5),
        ("V(2)", "X(2)", 0.5),
        ("X(1)", "T(1)", 0.5),
        ("X(1)", "V(1)", 0.5),
        ("X(2)", "R(2)", 1.0 / 3.0),
        ("X(2)", "M(2)", 1.0 / 3.0),
        ("X(2)", "#", 1.0 / 3.0),
        ("R(2)", "R(2)", 1.0 / 3.0),
        ("R(2)", "M(2)", 1.0 / 3.0),
        ("R(2)", "#", 1.0 / 3.0),
        ("M(2)", "#", 1.0),
    ],
    "eos": "#",
}


FSG_G2_B = {
    "label": "[B] Mathews et al (1989)",
    "states": ["#", "W(1)", "W(2)", "S(1)", "S(2)", "S(3)", "P(1)", "P(2)", "P(3)", "N(1)", "N(2)", "Z(1)"],
    "alphabet": ["W", "S", "N", "P", "Z"],
    "start_states": ["W(1)", "N(1)"],
    "terminal_states": [
        "Z(1)",
    ],
    "transitions": [
        ("W(1)", "W(2)", 1.0 / 3.0),
        ("W(1)", "S(2)", 1.0 / 3.0),
        ("W(1)", "S(1)", 1.0 / 3.0),
        ("S(1)", "S(2)", 1.0),
        ("S(2)", "W(2)", 1.0),
        ("W(2)", "S(3)", 0.5),
        ("W(2)", "P(3)", 0.5),
        ("N(1)", "N(2)", 0.5),
        ("N(1)", "P(2)", 0.5),
        ("P(1)", "P(2)", 0.5),
        ("P(1)", "N(1)", 0.5),
        ("P(2)", "N(2)", 1.0),
        ("S(3)", "P(1)", 1.0 / 3.0),
        ("S(3)", "P(2)", 1.0 / 3.0),
        ("S(3)", "N(2)", 1.0 / 3.0),
        ("N(2)", "P(3)", 0.5),
        ("N(2)", "Z(1)", 0.5),
        ("P(3)", "S(3)", 0.5),
        ("P(3)", "Z(1)", 0.5),
        ("Z(1)", "#", 1.0),
    ],
    "eos": "#",
}


FSG_G3_C = {
    "label": "[C] Reber (1967)",
    "states": ["#", "T(1)", "T(2)", "V(1)", "V(2)", "X(1)", "X(2)", "P(1)", "P(2)", "S(1)", "S(2)"],
    "alphabet": ["T", "V", "X", "P", "S"],
    "start_states": ["T(1)", "V(1)"],
    "terminal_states": ["S(1)", "S(2)"],
    "transitions": [
        ("T(1)", "P(1)", 0.5),
        ("T(1)", "T(2)", 0.5),
        ("P(1)", "P(1)", 0.5),
        ("P(1)", "T(2)", 0.5),
        ("T(2)", "S(1)", 0.5),
        ("T(2)", "X(2)", 0.5),
        ("S(1)", "#", 1.0),
        ("X(2)", "X(1)", 0.5),
        ("X(2)", "V(2)", 0.5),
        ("V(1)", "X(1)", 0.5),
        ("V(1)", "V(2)", 0.5),
        ("X(1)", "X(1)", 0.5),
        ("X(1)", "V(2)", 0.5),
        ("V(2)", "P(2)", 0.5),
        ("V(2)", "S(2)", 0.5),
        ("P(2)", "X(2)", 0.5),
        ("P(2)", "S(1)", 0.5),
        ("S(2)", "#", 1.0),
    ],
    "eos": "#",
}

FSG_G4_D = {
    "label": "[D] Skosnik et al (2002)",
    "states": ["#", "P(1)", "P(2)", "X(1)", "X(2)", "J(1)", "J(2)", "V(1)", "V(2)", "T(1)", "T(2)", "H(1)", "H(2)"],
    "alphabet": ["P", "X", "J", "V", "T", "H"],
    "start_states": ["P(1)", "X(1)"],
    "terminal_states": ["V(2)", "J(2)"],
    "transitions": [
        ("P(1)", "H(1)", 0.5),
        ("P(1)", "J(1)", 0.5),
        ("H(1)", "J(2)", 0.5),
        ("H(1)", "P(2)", 0.5),
        ("X(1)", "V(1)", 0.5),
        ("X(1)", "T(1)", 0.5),
        ("J(1)", "T(1)", 0.5),
        ("J(1)", "V(1)", 0.5),
        ("V(1)", "T(2)", 0.5),
        ("V(1)", "H(2)", 0.5),
        ("T(1)", "X(2)", 0.5),
        ("T(1)", "V(2)", 0.5),
        ("T(2)", "H(1)", 0.5),
        ("T(2)", "J(1)", 0.5),
        ("P(2)", "T(2)", 0.5),
        ("P(2)", "H(2)", 0.5),
        ("H(2)", "X(2)", 0.5),
        ("H(2)", "V(2)", 0.5),
        ("X(2)", "J(2)", 0.5),
        ("X(2)", "P(2)", 0.5),
        ("V(2)", "#", 1.0),
        ("J(2)", "#", 1.0),
    ],
    "eos": "#",
}


FSG_G5_E = {
    "label": "[E] Meulemans & Ven der Linder (1997)",
    "states": [
        "#",
        "F(1)",
        "F(2)",
        "F(3)",
        "D(1)",
        "D(2)",
        "J(1)",
        "J(2)",
        "J(3)",
        "H(1)",
        "H(2)",
        "H(3)",
        "Q(1)",
        "Q(2)",
        "M(1)",
        "M(2)",
    ],
    "alphabet": ["F", "D", "J", "H", "Q", "M"],
    "start_states": ["F(1)", "D(1)"],
    "terminal_states": ["F(3)", "J(3)"],
    "transitions": [
        ("F(1)", "D(2)", 1.0),
        ("D(2)", "Q(1)", 0.5),
        ("D(2)", "M(1)", 0.5),
        ("D(1)", "J(1)", 0.5),
        ("D(1)", "H(1)", 0.5),
        ("J(1)", "D(2)", 1.0),
        ("H(1)", "Q(1)", 0.5),
        ("H(1)", "M(1)", 0.5),
        ("Q(1)", "F(2)", 0.5),
        ("Q(1)", "H(2)", 0.5),
        ("F(2)", "F(2)", 0.5),
        ("F(2)", "H(2)", 0.5),
        ("H(2)", "F(3)", 0.5),
        ("H(2)", "H(3)", 0.5),
        ("M(1)", "J(2)", 0.5),
        ("M(1)", "Q(2)", 0.5),
        ("H(3)", "Q(2)", 0.5),
        ("H(3)", "J(2)", 0.5),
        ("Q(2)", "Q(2)", 0.5),
        ("Q(2)", "J(2)", 0.5),
        ("J(2)", "M(2)", 0.5),
        ("J(2)", "J(3)", 0.5),
        ("F(3)", "#", 1.0),
        ("M(2)", "F(3)", 0.5),
        ("M(2)", "H(3)", 0.5),
        ("J(3)", "#", 1.0),
    ],
    "eos": "#",
}


FSG_G6_F = {
    "label": "[F] Conway & Christiansen (2006)",
    "states": [
        "#",
        "M(1)",
        "M(2)",
        "M(3)",
        "M(4)",
        "V(1)",
        "V(2)",
        "V(3)",
        "V(4)",
        "V(5)",
        "T(1)",
        "T(2)",
        "T(3)",
        "X(1)",
        "X(2)",
        "X(3)",
        "X(4)",
        "R(1)",
        "R(2)",
        "R(3)",
        "R(4)",
    ],
    "alphabet": ["M", "V", "T", "X", "R"],
    "start_states": ["M(1)", "V(1)"],
    "terminal_states": ["X(4)", "R(4)", "V(3)", "M(3)", "R(3)", "T(1)"],
    "transitions": [
        ("M(1)", "V(2)", 0.5),
        ("M(1)", "X(1)", 0.5),
        ("V(1)", "M(2)", 0.5),
        ("V(1)", "X(2)", 0.5),
        ("V(2)", "R(1)", 0.5),
        ("V(2)", "X(3)", 0.5),
        ("X(1)", "R(2)", 1.0 / 3.0),
        ("X(1)", "R(3)", 1.0 / 3.0),
        ("X(1)", "T(1)", 1.0 / 3.0),
        ("M(2)", "R(2)", 1.0 / 3.0),
        ("M(2)", "R(3)", 1.0 / 3.0),
        ("M(2)", "T(1)", 1.0 / 3.0),
        ("X(2)", "V(4)", 0.5),
        ("X(2)", "T(2)", 0.5),
        ("R(1)", "V(3)", 0.5),
        ("R(1)", "M(3)", 0.5),
        ("X(3)", "R(2)", 1.0 / 3.0),
        ("X(3)", "R(3)", 1.0 / 3.0),
        ("X(3)", "T(1)", 1.0 / 3.0),
        ("V(4)", "R(2)", 1.0 / 3.0),
        ("V(4)", "R(3)", 1.0 / 3.0),
        ("V(4)", "T(1)", 1.0 / 3.0),
        ("T(2)", "X(4)", 0.5),
        ("T(2)", "R(4)", 0.5),
        ("X(4)", "#", 1.0),
        ("R(4)", "R(4)", 1.0 / 3.0),
        ("R(4)", "X(4)", 1.0 / 3.0),
        ("R(4)", "#", 1.0 / 3.0),
        ("V(3)", "V(3)", 1.0 / 3.0),
        ("V(3)", "M(3)", 1.0 / 3.0),
        ("V(3)", "#", 1.0 / 3.0),
        ("M(3)", "#", 1.0),
        ("R(3)", "V(3)", 1.0 / 3.0),
        ("R(3)", "M(3)", 1.0 / 3.0),
        ("R(3)", "#", 1.0 / 3.0),
        ("T(1)", "R(4)", 0.5),
        ("T(1)", "#", 0.5),
        ("T(3)", "T(3)", 1.0 / 3.0),
        ("T(3)", "M(4)", 1.0 / 3.0),
        ("T(3)", "V(5)", 1.0 / 3.0),
        ("V(5)", "X(2)", 0.5),
        ("V(5)", "M(2)", 0.5),
        ("M(4)", "V(2)", 0.5),
        ("M(4)", "X(1)", 0.5),
        ("R(2)", "V(5)", 1.0 / 3.0),
        ("R(2)", "M(4)", 1.0 / 3.0),
        ("R(2)", "T(3)", 1.0 / 3.0),
    ],
    "eos": "#",
}

FSG_G7_G = {
    "label": "[G] Knowlton & Squire (1996)",
    "states": ["#", "X(1)", "X(2)", "V(1)", "V(2)", "T(1)", "T(2)", "J(1)", "J(2)"],
    "alphabet": ["X", "V", "T", "J"],
    "start_states": ["X(1)", "V(1)"],
    "terminal_states": ["J(1)", "T(1)", "V(2)", "X(2)", "J(2)"],
    "transitions": [
        ("X(1)", "X(1)", 0.5),
        ("X(1)", "V(1)", 0.5),
        ("V(1)", "J(1)", 1.0 / 3.0),
        ("V(1)", "T(1)", 1.0 / 3.0),
        ("V(1)", "X(2)", 1.0 / 3.0),
        ("J(1)", "#", 0.5),
        ("J(1)", "T(2)", 0.5),
        ("T(2)", "X(1)", 0.5),
        ("T(2)", "V(1)", 0.5),
        ("T(1)", "#", 0.5),
        ("T(1)", "V(2)", 0.5),
        ("V(2)", "#", 0.5),
        ("V(2)", "J(2)", 0.5),
        ("X(2)", "#", 0.5),
        ("X(2)", "J(2)", 0.5),
        ("J(2)", "J(2)", 0.5),
        ("J(2)", "#", 0.5),
    ],
    "eos": "#",
}

FSG_G8_H = {
    "label": "[H] Reber & Allen (1978)",
    "states": ["#", "S(1)", "S(2)", "S(3)", "S(4)", "X(1)", "X(2)", "X(3)", "K(1)", "K(2)", "K(3)", "K(4)", "K(5)"],
    "alphabet": ["S", "X", "K"],
    "start_states": ["S(1)", "K(1)"],
    "terminal_states": ["S(4)", "K(5)"],
    "transitions": [
        ("S(1)", "S(2)", 0.5),
        ("S(1)", "X(1)", 0.5),
        ("S(2)", "K(2)", 0.5),
        ("S(2)", "X(2)", 0.5),
        ("X(1)", "S(3)", 0.5),
        ("X(1)", "K(3)", 0.5),
        ("K(1)", "X(1)", 0.5),
        ("K(1)", "S(2)", 0.5),
        ("K(2)", "K(4)", 0.5),
        ("K(2)", "S(3)", 0.5),
        ("X(2)", "K(4)", 0.5),
        ("X(2)", "S(3)", 0.5),
        ("S(3)", "K(4)", 0.5),
        ("S(3)", "S(4)", 0.5),
        ("K(3)", "S(4)", 0.5),
        ("K(3)", "K(4)", 0.5),
        ("K(4)", "K(5)", 0.5),
        ("K(4)", "S(4)", 0.5),
        ("S(4)", "#", 1.0),
        ("K(5)", "#", 1.0),
    ],
    "eos": "#",
}

FSG_G9_I = {
    "label": "[B] Brooks & Vokey (1991)",
    "states": ["#", "R(1)", "R(2)", "R(3)", "R(4)", "R(5)", "Q(1)", "Q(2)", "Q(3)"],
    "alphabet": ["R", "Q"],
    "start_states": ["R(1)", "Q(1)"],
    "terminal_states": ["R(5)", "Q(3)"],
    "transitions": [
        ("R(1)", "R(2)", 0.5),
        ("R(1)", "Q(1)", 0.5),
        ("Q(1)", "R(2)", 0.5),
        ("Q(1)", "Q(2)", 0.5),
        ("R(2)", "R(3)", 0.5),
        ("R(2)", "Q(2)", 0.5),
        ("Q(2)", "R(4)", 0.5),
        ("Q(2)", "Q(3)", 0.5),
        ("R(3)", "R(4)", 0.5),
        ("R(3)", "Q(3)", 0.5),
        ("R(4)", "R(5)", 0.5),
        ("R(4)", "Q(3)", 0.5),
        ("R(5)", "#", 1.0),
        ("Q(3)", "#", 1.0),
    ],
    "eos": "#",
}

FSG_G10_J = {
    "label": "[J] Witt & Vinter (2011)",
    "states": ["#", "L(1)", "L(2)", "L(3)", "L(4)", "N(1)", "N(2)", "N(3)", "P(1)", "P(2)", "P(3)", "P(4)"],
    "alphabet": ["L", "N", "P"],
    "start_states": ["L(1)", "P(1)"],
    "terminal_states": ["L(4)", "N(3)", "P(4)"],
    "transitions": [
        ("L(1)", "L(2)", 0.5),
        ("L(1)", "N(1)", 0.5),
        ("N(1)", "L(2)", 0.5),
        ("N(1)", "P(2)", 0.5),
        ("P(1)", "N(1)", 0.5),
        ("P(1)", "L(2)", 0.5),
        ("L(2)", "N(2)", 0.5),
        ("L(2)", "P(2)", 0.5),
        ("N(2)", "L(3)", 0.5),
        ("N(2)", "P(3)", 0.5),
        ("P(2)", "N(2)", 0.5),
        ("P(2)", "L(3)", 0.5),
        ("L(3)", "L(4)", 0.5),
        ("L(3)", "P(3)", 0.5),
        ("P(3)", "N(3)", 0.5),
        ("P(3)", "P(4)", 0.5),
        ("L(4)", "#", 1.0),
        ("N(3)", "#", 1.0),
        ("P(4)", "#", 1.0),
    ],
    "eos": "#",
}

# Simple 1st order Markov chain
Markov_1st_Order = {
    "label": "1st Order Markov Chain",
    "alphabet": ["A", "B", "C", "D"],
    "states": ["#", "A", "B", "C", "D"],
    "start_states": ["A", "B"],
    "terminal_states": ["C", "D"],
    "transitions": [
        # From A: can go to B or C
        ("A", "B", 0.6),
        ("A", "C", 0.4),
        # From B: can go to A, C, or D
        ("B", "A", 0.4),
        ("B", "C", 0.3),
        ("B", "D", 0.3),
        # From C: can go to A, B, or terminate
        ("C", "A", 0.3),
        ("C", "B", 0.3),
        ("C", "#", 0.4),
        # From D: can go to A or terminate
        ("D", "A", 0.5),
        ("D", "#", 0.5),
    ],
    "eos": "#",
}

# 2nd order Markov chain - P(X_t | X_{t-1}, X_{t-2})
# Uses numeric indices to encode the 2-symbol history
# Index encoding: 0=start, 1=after_A, 2=after_B, 3=after_AA, 4=after_AB, 5=after_BA, 6=after_BB
Markov_2nd_Order = {
    "label": "2nd Order Markov Chain",
    "alphabet": ["A", "B"],
    "states": [
        "#",
        # Start states (no history)
        "A(0)", "B(0)",
        # After 1 symbol (1st position)
        "A(1)", "B(1)",  # After A
        "A(2)", "B(2)",  # After B
        # After 2 symbols (2nd+ positions) - true 2nd order
        "A(3)", "B(3)",  # After AA
        "A(4)", "B(4)",  # After AB
        "A(5)", "B(5)",  # After BA
        "A(6)", "B(6)",  # After BB
    ],
    "start_states": ["A(0)", "B(0)"],
    "terminal_states": ["A(3)", "B(3)", "A(4)", "B(4)", "A(5)", "B(5)", "A(6)", "B(6)"],
    "transitions": [
        # From start (emit first symbol, no history yet)
        ("A(0)", "A(1)", 0.6),  # Start with A, now have "A" history
        ("A(0)", "B(1)", 0.4),  # Start with A, emit B, now have "A" history
        ("B(0)", "A(2)", 0.3),  # Start with B, emit A, now have "B" history
        ("B(0)", "B(2)", 0.7),  # Start with B, emit B, now have "B" history
        
        # From 1-symbol history to 2-symbol history
        ("A(1)", "A(3)", 0.7),  # After A, emit A -> have "AA" history
        ("A(1)", "B(3)", 0.3),  # After A, emit B -> have "AA" history
        ("B(1)", "A(4)", 0.4),  # After A, emit A -> have "AB" history
        ("B(1)", "B(4)", 0.6),  # After A, emit B -> have "AB" history
        ("A(2)", "A(5)", 0.2),  # After B, emit A -> have "BA" history
        ("A(2)", "B(5)", 0.8),  # After B, emit B -> have "BA" history
        ("B(2)", "A(6)", 0.5),  # After B, emit A -> have "BB" history
        ("B(2)", "B(6)", 0.5),  # After B, emit B -> have "BB" history
        
        # 2nd-order transitions: P(X_t | X_{t-2}=A, X_{t-1}=A)
        ("A(3)", "A(3)", 0.8),  # AA context, emit A, stay in AA
        ("A(3)", "B(3)", 0.1),  # AA context, emit B, stay in AA
        ("A(3)", "#", 0.1),     # AA context, terminate
        ("B(3)", "A(4)", 0.3),  # AA context, emit A, move to AB
        ("B(3)", "B(4)", 0.6),  # AA context, emit B, move to AB
        ("B(3)", "#", 0.1),     # AA context, terminate
        
        # 2nd-order transitions: P(X_t | X_{t-2}=A, X_{t-1}=B)
        ("A(4)", "A(5)", 0.2),  # AB context, emit A, move to BA
        ("A(4)", "B(5)", 0.7),  # AB context, emit B, move to BA
        ("A(4)", "#", 0.1),     # AB context, terminate
        ("B(4)", "A(6)", 0.4),  # AB context, emit A, move to BB
        ("B(4)", "B(6)", 0.5),  # AB context, emit B, move to BB
        ("B(4)", "#", 0.1),     # AB context, terminate
        
        # 2nd-order transitions: P(X_t | X_{t-2}=B, X_{t-1}=A)
        ("A(5)", "A(3)", 0.6),  # BA context, emit A, move to AA
        ("A(5)", "B(3)", 0.3),  # BA context, emit B, move to AA
        ("A(5)", "#", 0.1),     # BA context, terminate
        ("B(5)", "A(4)", 0.3),  # BA context, emit A, move to AB
        ("B(5)", "B(4)", 0.6),  # BA context, emit B, move to AB
        ("B(5)", "#", 0.1),     # BA context, terminate
        
        # 2nd-order transitions: P(X_t | X_{t-2}=B, X_{t-1}=B)
        ("A(6)", "A(5)", 0.4),  # BB context, emit A, move to BA
        ("A(6)", "B(5)", 0.5),  # BB context, emit B, move to BA
        ("A(6)", "#", 0.1),     # BB context, terminate
        ("B(6)", "A(6)", 0.2),  # BB context, emit A, stay in BB
        ("B(6)", "B(6)", 0.7),  # BB context, emit B, stay in BB
        ("B(6)", "#", 0.1),     # BB context, terminate
    ],
    "eos": "#",
}
