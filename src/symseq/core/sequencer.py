# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors
"""
symbolic.py

Implements abstract/base class used to generate symbolic sequences.
"""

# standard imports
import numpy as np
from abc import ABC

# local imports
from ..utils.io import get_logger, save_pickle

logger = get_logger(__name__)


# ######################################################################################################################
class SymbolicSequencer(ABC):
    """
    Build patterned symbolic sequences.
    Contains the generic constructors to implement structured symbolic sequences
    """

    def __init__(
        self,
        label: str = "DefaultSymbolicSequencer",
        alphabet_size: int | None = None,
        alphabet: list[str] | None = None,
        rng: np.random.Generator | None = None,
        verbose: bool = True,
    ):
        """
        Initialize the SymbolicSequencer. Either alphabet or alphabet_size must be provided.

        Parameters
        ----------
        label : str, optional
            Sequencer name.
        alphabet : list of str, optional
            Unique symbols. Required if alphabet_size is not provided, and takes precedence over alphabet_size.
        alphabet_size : int, optional
            Number of unique symbols. Required only if alphabet is not provided, in which case the generated alphabet is
            a list of integers. If None, the alphabet_size is calculated from the provided alphabet. Defaults to None.
        rng : numpy.random.Generator, optional
            Seeded random number generator. Defaults to a new generator, causing non-reproducible sequences.
        verbose : bool, optional
            If True, logs the generation process. Defaults to True.
        """
        self.label = label

        if alphabet is not None:
            if alphabet_size is not None and alphabet_size != len(alphabet):
                raise ValueError("Alphabet size must be equal to the length of the provided alphabet")
            self.alphabet = sorted(alphabet)  # sort the list to ensure consistency
            self.alphabet_size = len(alphabet)
        else:
            if not isinstance(alphabet_size, int):
                raise ValueError("Alphabet size must be an integer if alphabet is not provided")
            self.alphabet = [str(i) for i in range(alphabet_size)]
            self.alphabet_size = alphabet_size

        if verbose:
            logger.info(f"Created SymbolicSequencer object with label {self.label} and alphabet {self.alphabet}")

        if rng is None:
            self.rng = np.random.default_rng()
            if verbose:
                logger.warning("SymbolicSequencer sequences will not be reproducible!")
        else:
            self.rng = rng

    @classmethod
    def from_sequence(cls, sequence: list[str], label: str = "DefaultSymbolicSequencer", eos: str = "#"):
        """
        Create a SymbolicSequencer object from a sequence and infer parameters from it.

        Parameters
        ----------
        sequence : list of str
            Sequence to use as a basis for the SymbolicSequencer.
        label : str, optional
            Label of the current task. Defaults to "DefaultSymbolicSequencer".
        eos : str, optional
            End-of-sentence marker, which will be ignored and not added to the alphabet. Defaults to '#'.

        Returns
        -------
        SymbolicSequencer
            A new SymbolicSequencer object.
        """
        alphabet = sorted(set(sequence) - {eos})

        # Create and return a new SymbolicSequencer object
        return cls(label=label, alphabet=alphabet)

    def generate_random_sequence(self, length: int = 10, replace: bool = True, verbose: bool = True) -> list[str]:
        """
        Randomly draw items from the alphabet, where grammaticality obviously does not play a role.

        Parameters
        ----------
        seq_len : int, optional
            Length of the random sequence to generate. Defaults to 10.
        replace : bool, optional
            Whether to allow replacement of symbols in the random sequence. Defaults to True.
        verbose : bool, optional
            If True, logs information about the random sequence generation.

        Returns
        -------
        list[str] : A list containing the generated random sequence.

        """
        if verbose:
            logger.info(f"Generating a random sequence of length {length} from a set of {self.alphabet_size} symbols")
        alphabet_arr = np.array(self.alphabet, dtype=object)  # avoid conversion to np.str_ type
        random_sequence = list(self.rng.choice(alphabet_arr, length, replace=replace))
        return random_sequence

    def save(self, file_name=None, file_path=None):
        """
        Save the current SymbolicSequencer object to a file.

        Parameters
        ----------
        file_name : str, optional
            Name of the file to save. If None, the file name will be generated based on the label. Defaults to None.
        file_path : str, optional
            Path to the directory where the file should be saved. If None, the file will be saved in the current
            working directory. Defaults to None.

        """
        if file_name is None:
            file_name = f"SymbolicSequencer_{self.label}.pkl"

        save_pickle(self, file_name, file_path)

    # TODO plot distributions (string length, token frequency, ... string complexity), recurrence plots
