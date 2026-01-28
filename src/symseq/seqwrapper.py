# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""
seqwrapper.py

This module contains the SeqWrapper class and SeqData dataclass to provide a consistent interface
for subsequent processing.

"""

import os

# import toml
import toml
import yaml
from dataclasses import dataclass

# internal imports
from symseq.grammars.ag import ArtificialGrammar
from symseq.grammars.nad import NonAdjacentDependencies
from symseq.utils.io import get_logger
from symseq.utils.strtools import clean_string_set, concatenate_string_set
from symseq import tasks

logger = get_logger(__name__)


@dataclass
class SeqContainer:
    """
    Data class for storing and generating datasets.
    """

    # config: dict  # dataset section of the configuration file
    mode: str  # offline | online
    params: dict  # dataset parameters

    n_samples_train: int
    n_samples_test: int

    # train_set: list[list[str]]  # train sequence of symbols
    # test_set: list[list[str]]  # test sequence of symbols
    # train_states: list[list[str]]  # train sequence of (indexed) states
    # test_states: list[list[str]]  # test sequence of (indexed) states
    train_set: list[str]  # train sequence of symbols
    test_set: list[str]  # test sequence of symbols
    train_states: list[str]  # train sequence of (indexed) states
    test_states: list[str]  # test sequence of (indexed) states

    grammaticality_train: list[bool]  # train sequence of booleans
    grammaticality_test: list[bool]  # test sequence of booleans

    length_range: tuple[int, int]  # min, max length of sequences

    tasks: dict | None = None  # task targets


class SeqWrapper:
    """ """

    def __init__(self, config: dict):
        self.config = config
        self.generator = self._parse_generator_object(config["generator"])
        self.dataset = None

        # load dataset if provided
        if "dataset" in config:
            self._validate_dataset_config(config["dataset"])
            self.dataset = self.generate_data(**config["dataset"])

    @classmethod
    def from_config(cls, filepath):
        """
        Load a configuration file, parse it, and create a SeqWrapper object.

        Parameters
        ----------
        filepath : str
            Path to the configuration file.

        Returns
        -------
        SeqWrapper
            The SeqWrapper object.
        """
        with open(filepath) as f:
            config_data = yaml.load(f, Loader=yaml.FullLoader)
        config = cls._parse_config(config_data)
        return cls(config)

    @classmethod
    def from_dict(cls, config):
        """
        Create a SeqWrapper object from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary.

        Returns
        -------
        SeqWrapper
            The SeqWrapper object.
        """

        config = cls._parse_config(config)
        return cls(config)

    def _parse_generator_object(self, config_generator):
        """
        Parse the generator section of the configuration file and return a generator object.
        """
        if "type" not in config_generator:
            raise ValueError("Invalid configuration file. Missing 'generator.type' field.")

        if config_generator["type"] == "ArtificialGrammar":
            if "mode" not in config_generator:
                raise ValueError("Invalid configuration file. Missing 'generator.mode' field.")
            if "params" not in config_generator:
                raise ValueError("Invalid configuration file. Missing 'generator.params' field.")

            # meta-parameters
            seed = config_generator.get("seed", None)

            if config_generator["mode"] == "preset":
                g = ArtificialGrammar.from_preset(seed=seed, preset_name=config_generator["preset"])
            elif config_generator["mode"] == "random":
                g = ArtificialGrammar.from_constraints(seed=seed, **config_generator["constraints"])
            elif config_generator["mode"] == "custom":
                g = ArtificialGrammar(seed=seed, **config_generator["params"])
            else:
                raise ValueError(
                    f"Invalid generator mode: {config_generator['mode']}. Valid modes are 'preset', 'random', and 'custom'."
                )
            return g
        elif config_generator["type"] == "NonAdjacentDependencies":
            seed = config_generator.get("seed", None)

            g = NonAdjacentDependencies(seed=seed, **config_generator["params"])
            return g

        raise ValueError(f"Invalid generator type: {config_generator['type']}")

    @classmethod
    def _parse_config(cls, config) -> dict:
        """
        Parse the configuration file and extract the SymSeq parameters into a dictionary.
        """
        if "symseq" not in config:
            raise ValueError("Invalid configuration file. Missing 'symseq' section.")

        config = config["symseq"]

        if "generator" not in config:
            raise ValueError("Invalid configuration file. Missing 'generator' section.")
        # generator = cls._parse_generator_object(config["generator"])

        # load parameters from file
        if "params_file" in config["generator"]:
            params_file = config["generator"]["params_file"]
            if not os.path.exists(params_file):
                raise ValueError(f"Parameters file {params_file} does not exist.")

            # YAML file
            if params_file.endswith(".yaml"):
                with open(params_file) as f:
                    grammar_pars = yaml.load(f, Loader=yaml.FullLoader)
                    try:
                        config["generator"]["params"] = grammar_pars["generator"]["params"]
                    except KeyError:
                        raise ValueError("Invalid parameters file. Missing 'generator.params' section.")
            if params_file.endswith(".toml"):
                with open(params_file) as f:
                    grammar_pars = toml.load(f)
                    try:
                        config["generator"]["params"] = grammar_pars["generator"]["params"]
                    except KeyError:
                        raise ValueError("Invalid parameters file. Missing 'generator.params' section.")
            # Python file
            elif params_file.endswith(".py"):
                with open(params_file) as f:
                    grammar_pars = eval(f.read())
                    try:
                        config["generator"]["params"] = grammar_pars["generator"]["params"]
                    except KeyError:
                        raise ValueError("Invalid parameters file. Missing 'generator.params' section.")

        # load parameters/constraints from the YAML config file
        elif "params" in config["generator"] or "constraints" in config["generator"]:
            pass  # already loaded in generator.params section
        else:
            logger.warning("No parameters/constraints/file specified. Using default parameters for generator.")
            config["generator"]["params"] = {}

        return config

    def _validate_dataset_config(self, config):
        """
        Parse the dataset section of the configuration file and return a dataset object.

        Parameters
        ----------
        config : dict
            Dataset configuration under `symseq.dataset`.
        """
        if "params" not in config:
            raise ValueError("Invalid configuration file. Missing 'dataset.params' section.")
        if "train" not in config["params"]:
            raise ValueError("Invalid configuration file. Missing 'dataset.params.train' section.")
        if "test" not in config["params"]:
            raise ValueError("Invalid configuration file. Missing 'dataset.params.test' section.")

        params = config["params"]
        if "n_samples" not in params["train"]:
            raise ValueError("Invalid configuration file. Missing 'dataset.params.train.n_samples' field.")
        if "n_samples" not in params["test"]:
            raise ValueError("Invalid configuration file. Missing 'dataset.params.test.n_samples' field.")

    def _parse_tasks(self, config: dict, seqdata: SeqContainer) -> dict:
        """
        Parse the tasks section of the configuration file and return a dictionary of task targets.
        """
        task_targets = {}

        if "recognition_default" in config and config["recognition_default"] is True:
            train_outputs = tasks.generate_default_outputs(seqdata.train_set)
            test_outputs = tasks.generate_default_outputs(seqdata.test_set)

            # TODO extend to arbitrary tasks
            assert len(train_outputs) == len(test_outputs) == 1, "Only classification supported for now"
            assert train_outputs[0]["label"] == "classification", "Only classification supported for now"

            task_targets["classification"] = {
                "train": {
                    "labels": train_outputs[0]["output"],
                    "accept": train_outputs[0]["accept"],
                },
                "test": {
                    "labels": test_outputs[0]["output"],
                    "accept": test_outputs[0]["accept"],
                },
            }

        if "context_resolution" in config:
            t, a = tasks.classification(seqdata.train_states)
            task_targets["context_resolution"] = {"train": {"labels": t, "accept": a}}
            t, a = tasks.classification(seqdata.test_states)
            task_targets["context_resolution"]["test"] = {"labels": t, "accept": a}

        if "grammaticality_classification" in config:
            t, a = tasks.grammaticality_classification(
                seqdata.train_set, seqdata.grammaticality_train, eos=self.generator.eos
            )
            task_targets["grammaticality_classification"] = {"train": {"labels": t, "accept": a}}
            t, a = tasks.grammaticality_classification(
                seqdata.test_set, seqdata.grammaticality_test, eos=self.generator.eos
            )
            task_targets["grammaticality_classification"]["test"] = {"labels": t, "accept": a}

        return task_targets

    def generate_data(self, mode: str, params: dict, concat_strings: bool = True) -> SeqContainer:
        """
        Generate train and test datasets using the provided generator.

        Parameters
        ----------
        mode : str
            Generation mode (offline or online).
        params : dict
            Dataset generation parameters.

        Returns
        -------
        seqdata : SeqContainer
            A SeqContainer object containing the generated datasets (input and tasks).
        """
        logger.info(f"Generating datasets for generator {self.generator.label}...")

        if mode == "offline":
            # generate strings of states first
            train_states, gramm_train = self.generator.generate_string_set(as_states=True, **params["train"])
            # remove state indices, convert to symbols
            train_set = clean_string_set(train_states, eos=self.generator.eos, as_states=False)
            # concatenate into single list
            if concat_strings:
                train_states = concatenate_string_set(train_states, eos=self.generator.eos)
                train_set = concatenate_string_set(train_set, eos=self.generator.eos)

            # same for test set
            test_states, gramm_test = self.generator.generate_string_set(as_states=True, **params["test"])
            test_set = clean_string_set(test_states, eos=self.generator.eos, as_states=False)
            if concat_strings:
                test_states = concatenate_string_set(test_states, eos=self.generator.eos)
                test_set = concatenate_string_set(test_set, eos=self.generator.eos)

            seqdata = SeqContainer(
                mode=mode,
                params=params,
                n_samples_train=len(train_set),
                n_samples_test=len(test_set),
                train_set=train_set,
                test_set=test_set,
                train_states=train_states,
                test_states=test_states,
                grammaticality_train=gramm_train,
                grammaticality_test=gramm_test,
                length_range=params["train"].get("length_range", None),
            )

            # seqdata.tasks = self._parse_tasks(config["tasks"], seqdata)

        elif dataset_config.mode == "online":
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid dataset mode: {dataset_config.mode}")

        return seqdata

    @classmethod
    def from_args(cls, **kwargs):
        raise NotImplementedError

        # return cls(sequence, config=kwargs)
