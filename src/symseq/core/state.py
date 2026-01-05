# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors
"""
state.py

Defines the State class for representing states in a grammar.
"""

# standard imports
import re

# local imports
from ..utils.io import get_logger

logger = get_logger(__name__)


class State:
    @classmethod
    def from_string(cls, string):
        """
        Create a State object from a string. Metadata such as start/terminal are not included.

        Parameters
        ----------
        s : str
            The string to create the State object from. Expected string format is "SymbolName" or "SymbolName(index)",
            where SymbolName is the name of the symbol and index is the index of the state in the FSM.
            For example, "A(1)", "B (2)" or "C" are all valid state names.

        Returns
        -------
        State
            A new State object.
        """
        # return State(string)
        match_index = re.match(r"^(.*)\((\d+)\)", string)  # match SymbolName(index)

        if match_index:
            # Extract the parts before the parenthesis and the number inside the parenthesis
            symbol_name = match_index.group(1)
            index = int(match_index.group(2))
        else:  # no index provided, use the whole state label as symbol name
            symbol_name = string
            index = None

        return State(symbol_name, index)  # create a State object

    def __init__(self, symbol: str, index: int | None, start: bool = False, terminal: bool = False):
        """
        Initialize a State object.

        Parameters
        ----------
        symbol : str
            The name/label of the state.
        index : int or None
            The index of the state.
        start : bool, optional
            Whether the state is a start state. Defaults to False.
        terminal : bool, optional
            Whether the state is a terminal state. Defaults to False.
        """
        self.symbol = symbol
        self.index = index
        self.start = start
        self.terminal = terminal

    def __repr__(self):
        return f"State(symbol={self.symbol!r}, index={self.index}, start={self.start}, terminal={self.terminal})"

    def __str__(self):
        if self.index is not None:
            return f"{self.symbol}({self.index})"
        else:
            return f"{self.symbol}"

    def __eq__(self, other):
        if isinstance(other, State):
            return self.symbol == other.symbol and self.index == other.index
        elif isinstance(other, str):
            # print(f"Comparing {self} with {other}: {str(self) == other}")
            return str(self) == other
        else:
            raise TypeError(f"Cannot compare State with {type(other)}")

    def __hash__(self):
        return hash(self.tostring())

    def tostring(self):
        return str(self)
