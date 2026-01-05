# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Grammar inference and parsing."""

import warnings

try:
    from nltk import CFG
    from nltk.parse import ChartParser
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


def cyk_parse(sequence: list[str], grammar: str | None = None, start_symbol: str = 'S') -> dict:
    """
    Parse sequence using CYK algorithm via NLTK.

    Parameters
    ----------
    sequence : list of str
        Input sequence to parse.
    grammar : str, optional
        Context-free grammar in NLTK format.
        Example: "S -> A B\\nA -> 'a'\\nB -> 'b'"
    start_symbol : str, default='S'
        Start symbol of the grammar.

    Returns
    -------
    dict
        Parse results with keys:
        - 'recognized': bool, whether sequence is recognized
        - 'num_parses': int, number of valid parse trees
        - 'parses': list of parse trees (if recognized)

    Notes
    -----
    Uses NLTK's ChartParser which implements CYK-like bottom-up parsing.

    Grammar should be in NLTK CFG format:
    - Productions: "LHS -> RHS"
    - Terminals in quotes: 'a', 'b'
    - Non-terminals without quotes: S, A, B

    References
    ----------
    Cocke, J., & Schwartz, J. T. (1970). Programming languages
    and their compilers.

    Younger, D.H. (1967). Recognition and parsing of context-free
    languages in time n³. Information and Control, 10(2), 189-208.

    Examples
    --------
    >>> grammar_str = "S -> A B\\nA -> 'a'\\nB -> 'b'"
    >>> result = cyk_parse(['a', 'b'], grammar=grammar_str)
    >>> result['recognized']
    True
    """
    if not NLTK_AVAILABLE:
        raise ImportError(
            "NLTK is required for CYK parsing. "
            "Install with: pip install nltk"
        )

    if grammar is None:
        raise ValueError("Grammar must be provided")

    try:
        cfg = CFG.fromstring(grammar)
        parser = ChartParser(cfg)

        parses = list(parser.parse(sequence))

        return {
            'recognized': len(parses) > 0,
            'num_parses': len(parses),
            'parses': parses,
        }

    except Exception as e:
        warnings.warn(f"Parsing failed: {str(e)}", UserWarning)
        return {
            'recognized': False,
            'num_parses': 0,
            'parses': [],
            'error': str(e),
        }


def scfg_inference(sequences: list[list[str]], max_iter: int = 100) -> dict:
    """
    Infer Stochastic Context-Free Grammar via Inside-Outside algorithm.

    Parameters
    ----------
    sequences : list of list of str
        Training sequences.
    max_iter : int, default=100
        Maximum EM iterations.

    Returns
    -------
    dict
        Inferred SCFG with production probabilities.

    Notes
    -----
    This is a placeholder for future implementation.

    The Inside-Outside algorithm is the EM algorithm for SCFGs:
    - E-step: Compute inside/outside probabilities
    - M-step: Re-estimate production probabilities

    Full implementation requires:
    - Grammar initialization strategy
    - Inside/outside probability computation
    - Production probability re-estimation
    - Convergence criteria

    References
    ----------
    Lari, K. & Young, S.J. (1990). The estimation of stochastic
    context-free grammars using the Inside-Outside algorithm.
    Computer Speech & Language, 4(1), 35-56.
    """
    warnings.warn(
        "SCFG inference is not yet implemented. "
        "This is a complex algorithm requiring substantial development.",
        UserWarning
    )
    raise NotImplementedError(
        "SCFG inference via Inside-Outside algorithm requires full implementation. "
        "Consider using specialized libraries like NLTK or custom EM implementation."
    )
