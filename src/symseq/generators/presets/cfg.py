# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""
cfg.py

Preset context-free grammars for :class:`~symseq.generators.cfg.CFGGenerator`,
loadable via ``CFGGenerator.from_preset(name)``.

Each preset is a module-level dict of ``CFGGenerator.__init__`` kwargs (``label``,
``grammar``, and optional bounds). Grammars are written in NLTK PCFG notation.

Design constraints (so they behave well with the top-down sampler and the
chart-parser oracle):

- **No left recursion** — the sampler expands the leftmost symbol top-down, so
  left-recursive rules (``E -> E '+' E``) would blow past ``max_depth``. Recursive
  rules are written right-recursive, and every production of a recursive
  nonterminal consumes a terminal early.
- **No epsilon productions** — NLTK's chart parser is finicky with empty RHS, so
  every preset has a minimum non-empty string.
- **Probabilities tuned subcritical** — recursive productions carry low enough
  probability that the expected string length is finite.

"""

from __future__ import annotations

# {a^n b^n : n >= 1} — the canonical "counting" CFG; the textbook witness that
# context-free strictly contains regular.
anbn = {
    "label": "anbn",
    "grammar": "S -> 'a' S 'b' [0.5] | 'a' 'b' [0.5]",
    "max_length": 60,
}

# Even-length palindromes / mirror language w·reverse(w) over {a, b}. The
# "mirror" supra-regular grammar from AGL studies (Fitch & Hauser; de Vries).
palindrome = {
    "label": "palindrome",
    "grammar": (
        "S -> 'a' S 'a' [0.25] | 'b' S 'b' [0.25] | 'a' 'a' [0.25] | 'b' 'b' [0.25]"
    ),
    "max_length": 60,
}

# Nested matched dependencies over two pair types (open a/b, close A/B):
# e.g. a b B A. The context-free half of the nested-vs-cross-serial AGL contrast.
nested_dependencies = {
    "label": "nested_dependencies",
    "grammar": (
        "S -> 'a' S 'A' [0.25] | 'b' S 'B' [0.25] | 'a' 'A' [0.25] | 'b' 'B' [0.25]"
    ),
    "max_length": 60,
}

# Dyck-1: balanced single brackets. Every production starts with '(' (a terminal),
# so the top-down sampler always consumes before recursing.
dyck1 = {
    "label": "dyck1",
    "grammar": (
        "S -> '(' S ')' S [0.2] | '(' ')' S [0.2] | '(' S ')' [0.2] | '(' ')' [0.4]"
    ),
    "max_length": 60,
}

# Dyck-2: balanced strings over two bracket types, () and [].
dyck2 = {
    "label": "dyck2",
    "grammar": ( 
        "S -> '(' S ')' S [0.1] "
            "| '[' S ']' S [0.1] "
            "| '(' ')' S [0.1] "
            "| '[' ']' S [0.1] "
            "| '(' S ')' [0.1] "
            "| '[' S ']' [0.1] "
            "| '(' ')' [0.2] "
            "| '[' ']' [0.2]"
    ),
    "max_length": 60,
}

# Arithmetic expressions over 'id' with + and *. Right-recursive, unambiguous
# (precedence-encoding) form; verified subcritical so length stays finite.
arith = {
    "label": "arith",
    "grammar": (
        "E -> T '+' E [0.2] | T [0.8]\n"
        "T -> F '*' T [0.2] | F [0.8]\n"
        "F -> '(' E ')' [0.1] | 'id' [0.9]"
    ),
    "max_length": 60,
}

# Regular control: (ab)^n, right-linear. Useful as the regular baseline against
# the supra-regular presets above.
abn = {
    "label": "abn",
    "grammar": "S -> 'a' 'b' S [0.5] | 'a' 'b' [0.5]",
    "max_length": 60,
}


PRESETS: tuple[str, ...] = (
    "anbn",
    "palindrome",
    "nested_dependencies",
    "dyck1",
    "dyck2",
    "arith",
    "abn",
)


def list_presets() -> list[str]:
    """Return the names of the available CFG presets."""
    return list(PRESETS)
