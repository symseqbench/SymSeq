# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""
agl_dataset.py

This module contains functions for generating balanced AGL (Artificial Grammar Learning) datasets.
"""

from collections import defaultdict
from multiprocessing import cpu_count
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# internal imports
from symseq.utils.io import get_logger
from symseq.grammars.ag.utils import all_paths_as_strings, process_feature


logger = get_logger(__name__)


# TODO consider NSGA-II for multi-objective partitioning
# TODO add option for specified user train_set
def generate_balanced_agl(
    G,
    n_train_samples,
    factors: dict,
    factor_cells: list[dict],
    length_range=(1, 1000),
    n_deviants: int = 1,
    max_strings: int = 500,
    sampling: tuple[str, dict] = ("uniform", {}),
    # enforce_disjoint: bool = True,  # train and test sets must be disjoint
    n_proc=4,
    max_iter=int(1e4),
    verbose=False,
) -> tuple[list[list[str]], list[pd.DataFrame]]:
    """
    Generate a balanced AGL (Artificial Grammar Learning) string set.

    Parameters
    ----------
    G : ArtificialGrammar
        Grammar instance.
    n_train_samples : int
        Number of training samples.
    factors : dict
        Factors to use for the AGL dataset, with each key-value pair representing a factor and its levels and/or binning
        strategy.
    factor_cells : list[dict, int]
        Factor cells to use for the AGL dataset, with each dict representing a factor-level combination and the number
        of samples to generate for that cell.
    length_range : tuple, optional
        Length range for the strings. Defaults to (1, 1000).
    n_deviants : int, optional
        Number of deviants to introduce in the non-grammatical strings. Defaults to 1.
    max_strings : int, optional
        Maximum number of non-grammatical strings to generate. Defaults to 500.
    sampling : tuple[str, dict], optional
        Sampling strategy to use, specified as [sampling method, sampling kwargs]. Defaults to ("uniform", {}).
    n_proc : int, optional
        Number of processes to use for parallelization. Defaults to 4.
    verbose : bool, optional
        Whether to print progress. Defaults to False.

    Returns
    -------
    train_set : list of list of str
        List of strings for training.
    test_set : pd.DataFrame
        List of DataFrames with test set strings and features for each factor cell.
    """
    assert length_range[1] <= 20, "Length range is too large, consider using a smaller range (<20) for now"
    n_proc = min(n_proc or cpu_count(), cpu_count())  # number of processes to use
    n_proc = 1

    start_nodes = [str(s) for s in G.start_states]
    props = {
        "length_range": length_range,
        "eos": G.eos,
    }
    # generate all strings with constrained properties
    G_strs = all_paths_as_strings(G.graph, start_nodes, n_proc, **props)
    if verbose:
        logger.info(f"Computed {len(G_strs)} strings with valid property constraints.")
    G_strs = sorted(list(G_strs))

    G_strs = np.array(G_strs, dtype=object)
    G_strs = G.rng.permutation(G_strs)  # randomize string set

    # generate NG strings
    NG_strs = set()  # non-grammatical strings
    while len(NG_strs) < max_strings:
        NG_str = G.add_deviant(list(G.rng.choice(G_strs)), n_deviants=n_deviants)
        NG_strs.add(tuple(NG_str))
    NG_strs = np.array(sorted(list(NG_strs)), dtype=object)

    for cnt in range(max_iter):
        logger.info(f"Processing iteration {cnt}...")

        # select training strings
        train_set = G.rng.choice(G_strs, n_train_samples)

        # compute features for each string, in parallel
        str_feature_map = defaultdict(dict)  # string -> features & values
        feature_constraints = _build_feature_constraints(factors)  # build constraint/metric dict
        with ProcessPoolExecutor(max_workers=n_proc) as executor:
            strings = G_strs.tolist() + NG_strs.tolist()
            # map string <-> future(feature)
            futures = {executor.submit(process_feature, s, train_set, feature_constraints): s for s in strings}
            for f in as_completed(futures):
                results = f.result()
                string = futures[f]
                str_feature_map[string] |= results
        if verbose:
            logger.info(f"Computed features for {len(str_feature_map)} strings.")

        df = _to_dataframe(G_strs, NG_strs, str_feature_map)  # pipe strings+features to dataframe
        dfb = assign_factor_levels(df, factors)

        test_set = []
        error = False

        # 7) Stratified sampling (greedy; switch to ILP if tight)
        for cell_spec, k in factor_cells:
            try:
                picked = sample_for_specs(
                    df=dfb,  # candidates with metrics
                    spec=cell_spec,
                    k=k,
                    # stratify_col=stratify,
                    sampling=sampling,
                    replace=False,  # no duplicates
                    clamp=True,  # avoid oversampling tiny bins
                    random_state=7,
                    keep_label_col="condition",
                )
                test_set.append(picked)
            except Exception as e:
                error = True
                break
        if error:
            continue
        return train_set, test_set

    raise RuntimeError(f"Could not generate a balanced AGL set after {max_iter} tries.")


def _match_spec(df: pd.DataFrame, spec: dict) -> pd.Series:
    """
    Return a boolean mask for rows matching the given spec.
    Values in spec can be a single value, an iterable of allowed values,
    or a callable that returns bool per row.
    """
    mask = pd.Series(True, index=df.index)
    for col, wanted in spec.items():
        if callable(wanted):
            mask &= df[col].apply(wanted)
        elif isinstance(wanted, (list, tuple, set, pd.Series, np.ndarray)):
            mask &= df[col].isin(wanted)
        else:
            mask &= df[col] == wanted
    return mask


def sample_for_specs(
    df: pd.DataFrame,
    spec: dict,
    k: int,
    # stratify_col: list[str] | None = "length",
    sampling: tuple[str, dict] = ("uniform", {}),
    replace: bool = False,
    clamp: bool = True,
    random_state: int | None = 42,
    keep_label_col: str = "condition",
) -> pd.DataFrame:
    """
    For each spec (factor->level dict), filter df and sample k items.
    Optional length-stratified proportional sampling within each spec.
    """
    rng = np.random.default_rng(random_state)
    out = []

    spec = {f"F__{k}": v for k, v in spec.items()}
    stratify_cols = []
    # TODO improve this bit
    if sampling[0] == "stratified":
        assert "factors" in sampling[1]
        strat_factors = sampling[1]["factors"]
        for col in strat_factors:
            if col == "length":
                stratify_cols.append(col)
            else:
                stratify_cols.append(f"F__{col}")

    sub = df[_match_spec(df, spec)].copy()
    if sub.empty:
        raise ValueError(f"No rows match spec {spec}")

    if sampling[0] == "uniform":
        n = k if replace else min(k, len(sub))
        if not replace and k > len(sub):
            logger.warning(f"Not enough rows ({len(sub)}) to sample {k} without replacement for condition {spec}.")
        pick = sub.sample(n=n, replace=replace, random_state=int(rng.integers(0, 1_000_000)))
    else:
        # assert len(sampling[1]) == 1, "Only one feature stratification is currently supported"

        # verify that the stratification columns are in the dataframe
        assert all(c in sub.columns for c in sampling[1]["factors"]), "Stratification columns not in dataframe"

        # --- Stratified path ---
        if replace:
            stratify_col = sampling[1]["factors"][0]

            # Proportional with possible top-up (duplicates allowed)
            def per_bin(g):
                quota = int(np.ceil(k * len(g) / len(sub)))
                return g.sample(n=quota, replace=True, random_state=int(rng.integers(0, 1_000_000)))

            pick = sub.groupby(stratify_col, group_keys=False).apply(per_bin).head(k)
            if len(pick) < k:
                topup = sub.sample(n=k - len(pick), replace=True, random_state=int(rng.integers(0, 1_000_000)))
                pick = pd.concat([pick, topup], ignore_index=False)
        else:
            # Exact-k, no replacement via largest-remainder with capacity
            if len(sub) < k:
                logger.warning(f"Not enough rows ({len(sub)}) to sample {k} without replacement for condition {spec}.")
                k = min(k, len(sub))
                raise

            strata_sizes = sub.groupby(stratify_cols).size().sort_index()
            strata_keys = strata_sizes.index.to_list()
            n = strata_sizes.to_numpy(dtype=int)
            N = int(n.sum())

            # Ideal quotas
            q = k * n / N
            base = np.floor(q).astype(int)
            # Respect per-stratum capacity
            base = np.minimum(base, n)

            # Distribute remainders by fractional part, honoring capacity
            R = int(k - base.sum())
            if R > 0:
                frac = q - np.floor(q)
                cap = n - base
                # random jitter to break ties deterministically by seed
                jitter = np.zeros_like(frac, dtype=float)
                elig = np.where(cap > 0)[0]
                jitter[elig] = rng.random(len(elig)) * 1e-9
                order = np.argsort(-(frac + jitter))  # descending
                for idx in order:
                    if R == 0:
                        break
                    if cap[idx] > 0:
                        base[idx] += 1
                        cap[idx] -= 1
                        R -= 1

            # Sample per stratum without replacement
            picks = []

            for key, take in zip(strata_keys, base):
                if take <= 0:
                    continue
                block = sub.loc[sub.set_index(stratify_cols).index == key]
                picks.append(block.sample(n=take, replace=False, random_state=int(rng.integers(0, 1_000_000))))

            pick = pd.concat(picks, axis=0)

            # Shuffle final order and assert exact k
            if len(pick) != k:
                # If we somehow under-hit due to numerical edge cases, fill from residual strata
                residual = sub.drop(index=pick.index)
                needed = k - len(pick)
                if needed > 0 and len(residual) >= needed:
                    pick = pd.concat(
                        [pick, residual.sample(n=needed, replace=False, random_state=int(rng.integers(0, 1_000)))],
                        axis=0,
                    )
            pick = pick.sample(frac=1.0, random_state=int(rng.integers(0, 1_000)))
            assert len(pick) == k, f"Expected {k}, got {len(pick)}"

    tag = tuple(sorted(spec.items()))
    pick[keep_label_col] = [tag] * len(pick)
    out.append(pick)

    return pd.concat(out, ignore_index=False) if out else df.iloc[0:0]


def _filter_by_factors(df, factors):
    query_list = [f"F__{k} in {v['levels']}" for k, v in factors.items() if "levels" in v]
    query = " and ".join(query_list)
    return df.query(query)


def _build_feature_constraints(factors):
    feature_constraints = {}

    for k, v in factors.items():
        if k == "grammaticality":
            continue

        # TODO change default parameters for string similarity as per Bailey & Hahn 2001 - cost of 1 is assigned to
        # each substitution and a cost of 0.7 to each insertion or deletion (cf. Bailey & Hahn, 2001).
        if "kwargs" not in v:
            feature_constraints[k] = {"kwargs": {}}
        else:
            feature_constraints[k] = {"kwargs": v["kwargs"]}

    return feature_constraints


def assign_factor_levels(df, factors):
    out = []
    for _, sub in df.groupby("grammaticality"):
        for factor, factor_specs in factors.items():
            if factor == "grammaticality":
                p = {f"F__{factor}": np.where(sub[factor] == "G", "G", np.where(sub[factor] == "NG", "NG", "X"))}
                sub = sub.assign(**p)

            elif "binning" not in factor_specs:
                # default binning: equal ranges
                logger.warning(
                    f"No binning specified for factor {factor}. Defaulting to uniform binning (equal frequency)."
                )
                edges = np.linspace(0.0, 1.0, len(factor_specs["levels"]) + 1)
                ranges = zip(factor_specs["levels"], list(zip(edges[:-1], edges[1:])))
                conds = []
                labels = []
                for label, range_ in ranges:
                    low, high = bin_by_quantile(sub[factor], range_[0], range_[1])
                    logger.info(f"Binning {factor} uniformly by quantile ({low}, {high}) to label {label}")
                    conds += [(sub[factor] >= low) & (sub[factor] <= high)]
                    labels += [label]

                p = {f"F__{factor}": np.select(conds, labels, default="X")}
                sub = sub.assign(**p)

            elif factor_specs["binning"]["method"] == "quantile":
                conds = []
                labels = []
                for label, range_ in factor_specs["binning"]["ranges"].items():
                    low, high = bin_by_quantile(sub[factor], range_[0], range_[1])
                    logger.info(f"Binning {factor} by quantile ({low}, {high}) to label {label}")
                    conds += [(sub[factor] >= low) & (sub[factor] <= high)]
                    labels += [label]

                p = {f"F__{factor}": np.select(conds, labels, default="X")}
                sub = sub.assign(**p)
                # sub.head()

        out.append(sub)
    dfb = pd.concat(out)
    return dfb


def bin_by_quantile(series, q_low=0.25, q_high=0.75):
    """
    Bin a pandas Series by quantile.

    Parameters
    ----------
    series : pd.Series
        The Series to bin.
    q_low : float, optional
        The lower quantile to use. Defaults to 0.25.
    q_high : float, optional
        The upper quantile to use. Defaults to 0.75.

    Returns
    -------
    tuple[float, float]
        The lower and upper bounds of the bin.
    """
    lo, hi = series.quantile(q_low), series.quantile(q_high)
    return lo, hi


def _to_dataframe(G_strs, NG_strs, str_feature_map):
    """
    Build a DataFrame with features for the test set.

    Parameters
    ----------
    G_strs : list[tuple[str, bool]]
        List of grammatical strings.
    NG_strs : list[tuple[str, bool]]
        List of non-grammatical strings.
    str_feature_map : dict
        Dictionary mapping strings to feature values.

    Returns
    -------
    pd.DataFrame
        The DataFrame with features for the test set.
    """
    # build DAtaFrame with features for test set
    data_test = []
    concat_strings = list(zip(G_strs, [True] * len(G_strs))) + list(zip(NG_strs, [False] * len(NG_strs)))
    for string, grammatical in concat_strings:
        d = {
            "string": string,
            "length": len(string),
            "grammaticality": "G" if grammatical else "NG",
        } | str_feature_map[string]
        data_test.append(d)
    df = pd.DataFrame(data_test)
    return df
