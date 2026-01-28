# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""
nad.py

Contains classes for various non-adjacent dependency (NAD) models.
"""

import numpy as np
from more_itertools import collapse

# internal imports
from symseq.core.sequencer import SymbolicSequencer
from symseq.utils.io import get_logger, save_pickle

logger = get_logger(__name__)


class NonAdjacentDependencies(SymbolicSequencer):
    """
    Generate input and output sequences for tasks involving non-adjacent dependencies.

    Each input string consists of a frame of the type "A (n*X) B", where A and B are
    the dependents and X is the filler. `n` represents the span of the dependency
    (how many intervening items).

    References
    ----------
    [1] Fitz, H. (2011). A Liquid-State Model of Variability Effects in Learning
        Nonadjacent Dependencies. CogSci 2011 Proceedings, 897–902.
    [2] Lazar, A. (2009). SORN: a Self-organizing Recurrent Neural Network. Frontiers
        in Computational Neuroscience, 3(October), 23.
    [3] Onnis, ...
    """

    def __init__(
        self,
        label: str = "Default_NAD",
        n_deps: int | None = 1,
        n_unique_fillers: int | None = 1,
        dependency_pairs: list[tuple[str, str]] | None = None,
        fillers: list[str] | None = None,
        eos: str = "#",
        rng: np.random.Generator | None = None,
        seed: int | None = None,
        verbose: bool = True,
    ):
        """
        Initialize a NonAdjacentDependencies instance.

        Parameters
        ----------
        label : str
            Grammar label.
        n_deps : int
            Number of unique "words" or "frames" (A-B pairs). Each new frame is a novel dependency.
        n_unique_fillers : int
            Number of unique filler symbols.
        dependency_pairs : list of tuples, optional
            List of tuples containing the dependent element pairs with the structure (A1, B1). If None, the default
            is to generate the pairs based on the vocabulary size using pairs of the form (A1, B1).
        fillers : list of str, optional
            List of filler symbols. If None, the default is to generate the fillers based on the filler variability
            using symbols X1, X2, ...
        eos : str, optional
            End-of-string marker. Default is "#".
        rng : np.random.Generator, optional
            Random number generator instance for reproducibility. If None, a new
            default generator is created.
        verbose : bool, optional
            If True, logs the generation process. Default is True.
        """
        logger.info(f"Creating NonAdjacentDependencies instance...")

        self.label = label

        if rng is None:
            self.rng = np.random.default_rng(seed)
            if seed is None:
                logger.warning("NonAdjacentDependencies sequences will not be reproducible!")
        else:
            self.rng = rng

        # parse fillers
        if fillers is None:
            self.fillers = [f"X{i}" for i in range(n_unique_fillers)]
            self.n_unique_fillers = n_unique_fillers
        else:
            self.fillers = fillers
            self.n_unique_fillers = len(np.unique(np.array(self.fillers)))
            logger.info(f"Updated filler variability to {self.n_unique_fillers} based on provided fillers {fillers}.")
            assert len(self.fillers) == self.n_unique_fillers, "Provided fillers must be unique."

        # parse dependent elements
        # TODO include more detailed checks
        if dependency_pairs is None:
            self.dependency_pairs = [(f"A{i}", f"B{i}") for i in range(n_deps)]
            self.n_deps = n_deps
        else:
            self.dependency_pairs = dependency_pairs
            self.n_deps = len(set(dependency_pairs))
            logger.info(f"Updated vocabulary size to {self.n_deps} based on provided dependent elements.")
            assert len(self.dependency_pairs) == self.n_deps, "Provided dependent elements must be unique."

        self.vocabulary = self.generate_vocabulary(verbose=False)
        self.vocabulary_size = len(self.vocabulary)
        # TODO expected behavior for accepted patterns?
        # self.accepted_patterns = [f"A{i}B{i}" for i in range(vocabulary_size)]

        all_symbols = list(collapse(self.dependency_pairs + self.fillers))  # flatten list
        unique_symbols = list(np.unique(all_symbols))  # alphabet

        super().__init__(
            label=self.label,
            alphabet_size=len(unique_symbols),
            alphabet=unique_symbols,
            rng=self.rng,
        )

        self.start_symbols = [d[0] for d in self.dependency_pairs]
        self.terminal_symbols = [d[1] for d in self.dependency_pairs]
        self.eos = eos

        if verbose:
            self.print()

    def print(self):
        """
        Displays all the relevant information.
        """
        logger.info("***************************************************************************")
        logger.info(f"Non-adjacent dependency generator: {self.label}")
        logger.info(f"Alphabet: {self.alphabet}")
        logger.info(f"Start symbols: {self.start_symbols}")
        logger.info(f"Terminal symbols: {self.terminal_symbols}")
        logger.info(f"Fillers: {self.fillers}")

    # TODO consider adding generators
    def generate_string(
        self, filler_len: int = 1, randomize_fillers: bool = False, generator: bool = False
    ) -> list[str]:
        """
        Generate an individual string, 'word' or frame.

        Parameters
        ----------
        n_fillers : int, optional
            Number of fillers elements (dependency length) in the string. Default is 1.
        randomize_fillers : bool, optional
            Whether to randomize the fillers if multiple present (e.g., A1 X1 X2 B1). Default is False.
        generator : bool, optional
            Retrieve the string as a generator (True) or a list (False). Default is False.

        Returns
        -------
        list of str
            The generated string as a list of symbols.

        Examples
        --------
        >>> nAD = NonAdjacentDependencies(vocabulary_size=3, filler_variability=2, dependency_length=2)
        >>> nAD.generate_string()
        ['A2', 'X1', 'X1', 'B2']
        >>> nAD.generate_string(randomize_fillers=True)
        ['A1', 'X1', 'X0', 'B1']
        """
        # select a random dependent element pair
        d1, d2 = self.dependency_pairs[self.rng.integers(self.n_deps)]

        if len(self.fillers) > 0:
            if randomize_fillers:
                fillers = self.rng.choice(np.array(self.fillers, dtype=object), filler_len, replace=True).tolist()
            else:
                fillers = filler_len * [self.rng.choice(np.array(self.fillers, dtype=object))]  # choose one randomly
            string = [d1] + fillers + [d2]
        else:
            logger.warning("No fillers provided, using only the dependent element.")
            string = [d1, d2]

        if generator:
            raise NotImplementedError("Generator not implemented")
        else:
            return list(collapse(string))

    # TODO rename function
    def generate_vocabulary(self, filler_len: int | None = None, generator: bool = False, verbose: bool = True):
        """
        Generate the complete string set (all frames/words) for the provided set of dependent elements with `n_fillers`
        fillers. This can only be performed for non-randomized fillers.

        Parameters
        ----------
        n_fillers : int, optional
            Number of fillers elements (dependency length) in the string. If None, the internal `self.n_fillers` is
            used. Default is None.
        generator : bool, optional
            Retrieve the strings as generators (True) or lists (False). Default is False.

        Returns
        -------
        string_set : list of list of str
            The generated string set as a list of lists of symbols.

        """
        if verbose:
            logger.info(f"Generating the complete set of words/frames, according to {self.label} rules...")

        if filler_len is None:
            filler_len = self.n_unique_fillers

        string_set = []

        for d1, d2 in self.dependency_pairs:
            if len(self.fillers) > 0:
                for filler in self.fillers:
                    string = [d1] + filler_len * [filler] + [d2]
                    string_set.append(string)
            else:
                if verbose:
                    logger.warning("No fillers provided, using only the dependent element.")
                string = [d1, d2]
                string_set.append(string)

        if generator:
            raise NotImplementedError("Generator not implemented")
        else:
            return string_set

    # TODO consider adding option for introducing violations/deviants
    def generate_string_set(
        self,
        n_samples: int,
        filler_len: int | None = None,
        randomize_fillers: bool = False,
        frac_violations: float = 0.0,
        replace: bool = True,
        strict: bool = True,
        **kwargs,
    ):
        """
        Generate a string set of specified length for the experiment.

        Parameters
        ----------
        n_samples : int
            Total number of strings to generate.
        filler_len : int, optional
            Number of fillers elements (dependency length) in the string. Default is 1.
        randomize_fillers : bool, optional
            Whether to randomize the fillers if multiple present (e.g., A1 X1 X2 B1). Default is False.
        frac_violations : float, optional
            Introduce syntactic violations in this number of strings in the set.
        replace: bool
            Allow repetitions of the same string(s) in the generated set.
        strict: bool
            If True, raise an error when the vocabulary (unique strings) is exhausted and could not generate
            `n_samples` strings. If False, return all possible strings with warning.

        Returns
        -------
        string_set : list of list of str
            The generated string set as a list of lists of symbols.
        """
        if frac_violations > 0:
            raise NotImplementedError("This function is not implemented yet.")

        if not replace:
            if strict and n_samples > self.vocabulary_size:
                raise ValueError(
                    f"Cannot generate {n_samples} strings without replacement, vocabulary size is {self.vocabulary_size}"
                )

        logger.info(
            f"Generating {n_samples} strings with {frac_violations}% violations, according to {self.label} rules..."
        )

        string_set = []

        # generate all strings without violations, for now
        k = 0
        max_samples = n_samples if replace else self.vocabulary_size
        while k < max_samples:
            string = self.generate_string(filler_len, randomize_fillers)
            if not replace:
                if string not in string_set:
                    string_set.append(string)
                    k += 1
            else:
                string_set.append(string)
                k += 1

        # introduce violations
        self._apply_non_matching_frames(string_set, frac_violations)

        grammaticality = [True] * len(string_set)

        return string_set, grammaticality

    # TODO finish implementation
    def _apply_non_matching_frames(self, string_set: list[list[str]], frac_violations: float = 0.5):
        """
        Generate strings that violate the dependency by modifying the last (dependent) symbol in some strings.
        This function modifies the string set in place.

        TODO: currently assumes that the string_set only contains strings valid strings!!!

        Parameters
        ----------
        string_set : list of list of str
            The set of strings to process.
        frac_violations : float, optional
            Fraction of the string set to introduce violations in. Default is 0.5.

        Returns
        -------
        None
        """
        idx_ng_strings = self.rng.permutation(len(string_set))[: int(frac_violations * len(string_set))]

        for idx in idx_ng_strings:
            violating_terminals = sorted(list(set(self.terminal_symbols) - set([string_set[idx][-1]])))
            string_set[idx][-1] = str(self.rng.choice(violating_terminals))

    def save(self, file_name=None, file_path=None):
        """
        Save the current NonAdjacentDependencies object to a file.

        Parameters
        ----------
        file_name : str, optional
            Name of the file to save. If None, the file name will be generated based on the label. Defaults to None.
        file_path : str, optional
            Path to the directory where the file should be saved. If None, the file will be saved in the current
            working directory. Defaults to None.

        """
        if file_name is None:
            if self.label == "Default_nAD":
                file_name = f"{self.label}.pkl"
            else:
                file_name = f"nAD_{self.label}.pkl"

        save_pickle(self, file_name, file_path)


class CrossedNonAdjacentDependencies(SymbolicSequencer):
    """
    Generate input and output sequences for tasks involving crossed non-adjacent dependencies.
    Each input string consists of a frame of the type "A1 A2 A3 B1 B2 B3".

    References
    ----------
    """

    def __init__(
        self,
        label: str = "Default_Crossed_nAD",
        vocabulary_size: int = 1,
        dependent_elements: list[tuple[str, str]] | None = None,
        eos: str = "#",
        rng: np.random.Generator | None = None,
        verbose: bool = True,
    ):
        """
        Initialize a NonAdjacentDependencies instance.

        Parameters
        ----------
        vocabulary_size : int
            Number of dependent element pairs [(A1, B1), (A2, B2), ...].
        dependent_elements : list of tuples, optional
            List of tuples containing the dependent element pairs with the structure (A1, B1). If None, the default
            is to generate the pairs based on the vocabulary size using pairs of the form (A1, B1).
        eos : str, optional
            End-of-string marker. Default is "#".
        rng : np.random.Generator, optional
            Random number generator instance for reproducibility. If None, a new
            default generator is created.
        verbose : bool, optional
            If True, logs the generation process. Default is True.
        """
        logger.info(f"Creating CrossedNonAdjacentDependencies instance...")

        self.label = label

        if rng is None:
            self.rng = np.random.default_rng()
            logger.warning("NonAdjacentDependencies sequences will not be reproducible!")
        else:
            self.rng = rng

        # parse dependent elements  TODO this should probably include more detailed checks
        if dependent_elements is None:
            self.dependent_elements = [(f"A{i}", f"B{i}") for i in range(vocabulary_size)]
            self.vocabulary_size = vocabulary_size
        else:
            self.dependent_elements = dependent_elements
            self.vocabulary_size = len(np.unique(np.array(dependent_elements)))
            logger.info(f"Updated vocabulary size to {self.vocabulary_size} based on provided dependent elements.")
            assert len(self.dependent_elements) == self.vocabulary_size, "Provided dependent elements must be unique."

        unique_symbols = list(np.unique(list(collapse(self.dependent_elements))))  # alphabet
        unique_symbols = [str(s) for s in unique_symbols]

        super().__init__(
            label=self.label,
            alphabet_size=len(unique_symbols),
            alphabet=unique_symbols,
            rng=self.rng,
        )

        self.start_symbols = [d[0] for d in self.dependent_elements]
        self.terminal_symbols = [d[1] for d in self.dependent_elements]

        if verbose:
            self.print()

    def print(self):
        """
        Displays all the relevant information.
        """
        logger.info("***************************************************************************")
        logger.info(f"Crossed non-adjacent dependency generator: {self.label}")
        logger.info(f"Alphabet: {self.alphabet}")
        logger.info(f"Start symbols: {self.start_symbols}")
        logger.info(f"Terminal symbols: {self.terminal_symbols}")

    def generate_string(self, generator: bool = False) -> list[str]:
        """
        Generate an random string with crossed non-adjacent dependencies.

        Parameters
        ----------
        n_fillers : int, optional
            Number of fillers elements (dependency length) in the string. Default is 1.
        randomize_fillers : bool, optional
            Whether to randomize the fillers if multiple present (e.g., A1 X1 X2 B1). Default is False.
        generator : bool, optional
            Retrieve the string as a generator (True) or a list (False). Default is False.

        Returns
        -------
        list of str
            The generated string as a list of symbols.

        Examples
        --------
        """
        dep_start = []
        dep_end = []

        shuffled_idxs = self.rng.permutation(np.arange(self.vocabulary_size))
        for idx in shuffled_idxs:
            dep_start.append(self.dependent_elements[idx][0])
            dep_end.append(self.dependent_elements[idx][1])

        string = dep_start + dep_end

        if generator:
            raise NotImplementedError("Generator not implemented")
        else:
            return string

    # TODO consider adding option for introducing violations/deviants
    def generate_string_set(self, set_length: int, frac_violations: float = 0.0):
        """
        Generate a string set of specified length for the experiment.

        Parameters
        ----------
        set_length : int
            Total number of strings to generate.
        frac_violations : float, optional
            Introduce syntactic violations in this number of strings in the set.

        Returns
        -------
        string_set : list of list of str
            The generated string set as a list of lists of symbols.
        """
        if frac_violations > 0:
            raise NotImplementedError("This function is not implemented yet.")

        logger.info(
            f"Generating {set_length} strings with {frac_violations}% violations, according to {self.label} rules..."
        )
        string_set = []

        # generate all strings without violations, for now
        for _ in range(set_length):
            string = self.generate_string()
            string_set.append(string)

        # # introduce violations
        # self._non_matching_frames(string_set, frac_violations)

        return string_set

    def save(self, file_name=None, file_path=None):
        """
        Save the current CrossedNonAdjacentDependencies object to a file.

        Parameters
        ----------
        file_name : str, optional
            Name of the file to save. If None, the file name will be generated based on the label. Defaults to None.
        file_path : str, optional
            Path to the directory where the file should be saved. If None, the file will be saved in the current
            working directory. Defaults to None.

        """
        if file_name is None:
            if self.label == "Default_Crossed_nAD":
                file_name = f"{self.label}.pkl"
            else:
                file_name = f"Crossed_nAD_{self.label}.pkl"

        save_pickle(self, file_name, file_path)

    # def _non_matching_frames(self, string_set: list[list[str]], frac_violations: float = 0.5):
    #     """
    #     Generate strings that violate the dependency by modifying the last (dependent) symbol in some strings.
    #     This function modifies the string set in place.

    #     Parameters
    #     ----------
    #     string_set : list of list of str
    #         The set of strings to process.
    #     frac_violations : float, optional
    #         Fraction of the string set to introduce violations in. Default is 0.5.

    #     Returns
    #     -------
    #     None
    #     """
    #     idx_ng_strings = self.rng.permutation(len(string_set))[: int(frac_violations * len(string_set))]

    #     for idx in idx_ng_strings:
    #         violating_terminals = sorted(list(set(self.terminal_symbols) - set([string_set[idx][-1]])))
    #         string_set[idx][-1] = str(self.rng.choice(violating_terminals))


# TODO add fillers / noise
class NestedNonAdjacentDependencies(SymbolicSequencer):
    """
        Generate input and output sequences for tasks involving nested non-adjacent dependencies.
        Each input string consists of a frame of the type "A1 A2 A3 B3 B2 B1".
    A2 A1 A3 B3 B1 B2

        References
        ----------
    """

    def __init__(
        self,
        label: str = "Default_Nested_nAD",
        vocabulary_size: int = 1,
        dependent_elements: list[tuple[str, str]] | None = None,
        eos: str = "#",
        rng: np.random.Generator | None = None,
        verbose: bool = True,
    ):
        """
        Initialize a NestedNonAdjacentDependencies instance.

        Parameters
        ----------
        vocabulary_size : int
            Number of dependent element pairs [(A1, B1), (A2, B2), ...].
        dependent_elements : list of tuples, optional
            List of tuples containing the dependent element pairs with the structure (A1, B1). If None, the default
            is to generate the pairs based on the vocabulary size using pairs of the form (A1, B1).
        eos : str, optional
            End-of-string marker. Default is "#".
        rng : np.random.Generator, optional
            Random number generator instance for reproducibility. If None, a new
            default generator is created.
        verbose : bool, optional
            If True, logs the generation process. Default is True.
        """
        logger.info(f"Creating Nested NonAdjacentDependencies instance...")

        self.label = label

        if rng is None:
            self.rng = np.random.default_rng()
            logger.warning("Nested NonAdjacentDependencies sequences will not be reproducible!")
        else:
            self.rng = rng

        # parse dependent elements
        # TODO include more detailed checks
        if dependent_elements is None:
            self.dependent_elements = [(f"A{i}", f"B{i}") for i in range(vocabulary_size)]
            self.vocabulary_size = vocabulary_size
        else:
            self.dependent_elements = dependent_elements
            self.vocabulary_size = len(np.unique(np.array(dependent_elements))) // 2
            logger.info(f"Updated vocabulary size to {self.vocabulary_size} based on provided dependent elements.")
            assert len(self.dependent_elements) == self.vocabulary_size, "Provided dependent elements must be unique."

        unique_symbols = list(np.unique(list(collapse(self.dependent_elements))))  # alphabet
        unique_symbols = [str(s) for s in unique_symbols]

        super().__init__(
            label=self.label,
            alphabet_size=len(unique_symbols),
            alphabet=unique_symbols,
            rng=self.rng,
        )

        self.start_symbols = [d[0] for d in self.dependent_elements]
        self.terminal_symbols = [d[1] for d in self.dependent_elements]

        if verbose:
            self.print()

    def print(self):
        """
        Displays all the relevant information.
        """
        logger.info("***************************************************************************")
        logger.info(f"Crossed non-adjacent dependency generator: {self.label}")
        logger.info(f"Alphabet: {self.alphabet}")
        logger.info(f"Start symbols: {self.start_symbols}")
        logger.info(f"Terminal symbols: {self.terminal_symbols}")

    # TODO add option to not randomize string
    def generate_string(self, generator: bool = False) -> list[str]:
        """
        Generate an random string with nested non-adjacent dependencies.

        Parameters
        ----------
        generator : bool, optional
            Retrieve the string as a generator (True) or a list (False). Default is False.

        Returns
        -------
        list of str
            The generated string as a list of symbols.

        Examples
        --------
        """
        if generator:
            raise NotImplementedError("Generator not implemented")

        idxs_rand = self.rng.permutation(np.arange(self.vocabulary_size))

        # Create the ascending part
        first_half = [self.start_symbols[i] for i in idxs_rand]
        # Create the descending part in reverse order
        second_half = [self.terminal_symbols[i] for i in idxs_rand[::-1]]

        # Combine both parts into a single list
        return first_half + second_half

    # TODO consider adding option for introducing violations/deviants
    def generate_string_set(self, set_length: int, frac_violations: float = 0.0):
        """
        Generate a string set of specified length for the experiment.

        Parameters
        ----------
        set_length : int
            Total number of strings to generate.
        frac_violations : float, optional
            Introduce syntactic violations in this number of strings in the set.

        Returns
        -------
        string_set : list of list of str
            The generated string set as a list of lists of symbols.
        """
        if frac_violations > 0:
            raise NotImplementedError("This function is not implemented yet.")

        logger.info(
            f"Generating {set_length} strings with {frac_violations}% violations, according to {self.label} rules..."
        )
        string_set = []

        # generate all strings without violations, for now
        for _ in range(set_length):
            string = self.generate_string()
            string_set.append(string)

        # # introduce violations
        # self._non_matching_frames(string_set, frac_violations)

        return string_set

    def save(self, file_name=None, file_path=None):
        """
        Save the current NestedNonAdjacentDependencies object to a file.

        Parameters
        ----------
        file_name : str, optional
            Name of the file to save. If None, the file name will be generated based on the label. Defaults to None.
        file_path : str, optional
            Path to the directory where the file should be saved. If None, the file will be saved in the current
            working directory. Defaults to None.

        """
        if file_name is None:
            if self.label == "Default_Nested_nAD":
                file_name = f"{self.label}.pkl"
            else:
                file_name = f"Nested_nAD_{self.label}.pkl"

        save_pickle(self, file_name, file_path)

    # def _non_matching_frames(self, string_set: list[list[str]], frac_violations: float = 0.5):
    #     """
    #     Generate strings that violate the dependency by modifying the last (dependent) symbol in some strings.
    #     This function modifies the string set in place.

    #     Parameters
    #     ----------
    #     string_set : list of list of str
    #         The set of strings to process.
    #     frac_violations : float, optional
    #         Fraction of the string set to introduce violations in. Default is 0.5.

    #     Returns
    #     -------
    #     None
    #     """
    #     idx_ng_strings = self.rng.permutation(len(string_set))[: int(frac_violations * len(string_set))]

    #     for idx in idx_ng_strings:
    #         violating_terminals = sorted(list(set(self.terminal_symbols) - set([string_set[idx][-1]])))
    #         string_set[idx][-1] = str(self.rng.choice(violating_terminals))
