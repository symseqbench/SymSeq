# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""
ag_viz.py

Visualization functions for artificial grammars.

Author: Renato Duarte, Barna Zajzon
"""

from __future__ import annotations
import matplotlib
import matplotlib.axes
from sklearn.preprocessing import normalize

# from symseq.grammars.ag import ArtificialGrammar
from symseq.utils.strtools import chunk_transitions
from symseq.utils.io import get_logger
from symseq.viz.mc_graph import MarkovChain, Node

logger = get_logger(__name__)


def draw_graph(g, max_lift=1, save="./last.png"):
    P = g.transition_table(correct=True, verbose=True)
    mc = MarkovChain(P, g.states, title=r"$P_{s}$")
    mc.draw()

    for lift in range(max_lift):
        frequencies = chunk_transitions(g.generate_sequence(), lift + 1, return_labels=True)
        n_frequencies = normalize(frequencies, axis=1, norm="l1")

        mc = MarkovChain(n_frequencies, list(frequencies.columns), title=r"$P_{freq}$")
        label = save.split(".")[-2] + "_lift{}".format(lift) + save.split(".")[-1]
        mc.draw(label)


# TODO split into two functions, with / without lift
def plot_grammar(
    grammar: ArtificialGrammar,
    lift: int = 0,
    ax: matplotlib.axes.Axes | None = None,
    save: str | None = None,
    display: bool = False,
    **kwargs,
):
    """
    Plots a graph for a single case, specified by the parameter lift.

    Parameters
    ----------
    grammar : ArtificialGrammar
            The grammar to plot.
    lift: int
            The lift value for the grammar. TODO implement
    ax : matplotlib.Axis, optional
            The axis to plot on. If None, a new axis is created.
    save : str, optional
            The (absolute) path to save the figure to. If None, the figure is not saved.
    display : bool, optional
            Whether to display the figure. Defaults to False.
    kwargs : dict
            Additional keyword arguments to pass to the MarkovChain constructor.
    """
    transition_table = grammar.transition_table.T  # must transpose here for correct plotting

    if lift == 0:
        mc = MarkovChain(transition_table, grammar.states, **kwargs)
        mc.draw(grammar.terminal_states, grammar.eos, ax=ax, save=save)
    else:
        raise NotImplementedError("Lifting not implemented yet")
        frequencies = chunk_transitions(grammar.generate_sequence(), lift + 1, return_labels=True)
        n_frequencies = normalize(frequencies, axis=1, norm="l1")

        mc = MarkovChain(n_frequencies, list(frequencies.columns), title=r"$P_{freq}$", fontsize=28, node_fontsize=32)
        mc.draw(grammar.terminal_states, ax=ax)
        mc.draw(ax=ax)
