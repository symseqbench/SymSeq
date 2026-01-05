# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""
synthesis.py

This module generates regular grammars with a given number of nodes and a given target complexity.
"""

import numpy as np
import networkx as nx
import time as t
import copy

from symseq.utils.io import get_logger

logger = get_logger(__name__)

# TODO check if np.random needs to be initialized for cluster runs.
# if so:
# We do not recommend using small seeds below 32 bits for general use.
# Using just a small set of seeds to instantiate larger state spaces means that there are some initial states that are impossible to reach.
# This creates some biases if everyone uses such values.


# returns, in this order, (networkx) graph, count evolution snapshot array, score evolution snapshot array, computation time
def generate_grammar_with_target_te_complexity(
    target_complexity: float,  # in log space
    n_nodes: int,
    score_convergence_cutoff: float = 0.9999,
    sampling_interval: int = 5,
    n_iter_steps: int | None = None,  # write_test_output, document_motif_stats
    score_diff_inflation: float | None = None,
    with_density=False,
    p_mean=0.2,
    with_detailed_density=False,
    p_unidir=0.2,
    p_bidir=0.05,
    eps_bias=1e-12,
    rng: np.random.Generator | None = None,
) -> tuple[dict, dict]:
    time_start_total = t.time()

    if rng is None:
        rng = np.random.default_rng(42)

    # TODO make bias adaptive and smooth it out over time
    eps_bias = max(eps_bias, np.exp(target_complexity) * 1e-8)  # avoids log(0) scores in the beginning

    if n_iter_steps is None:
        n_iter_steps = n_nodes * (n_nodes - 1) * 10

    if score_diff_inflation is None:
        score_diff_inflation = 5 * (n_nodes) ** 2

    number_of_possible_edges = n_nodes * (n_nodes - 1) / 2

    if p_unidir and p_bidir:
        connection_probability = p_unidir + p_bidir
    else:
        connection_probability = p_mean

    logger.info(f"Target average connection probability: { p_mean}")

    # initial network generation
    edges = rng.poisson(connection_probability * number_of_possible_edges)  # / 0.6
    edge_indices = rng.choice(
        np.arange(n_nodes * n_nodes), edges, False
    )  # np.random.randint(0, par.number_of_nodes * par.number_of_nodes - 1, edges)
    edge_list = [
        [int(np.floor(i / n_nodes)), int(np.remainder(i, n_nodes))]
        for i in edge_indices
        if not (int(np.floor(i / n_nodes)) == int(np.remainder(i, n_nodes)))
    ]
    edge_list_bi = [[x[1], x[0]] for x in edge_list if rng.random() <= p_bidir / connection_probability]
    edge_list = edge_list + edge_list_bi

    g_plus = nx.DiGraph()
    g_plus.add_edges_from(edge_list)

    g_minus = nx.DiGraph()
    g_minus.add_edges_from(edge_list)

    logger.info("Finished initial network setup.")
    logger.info(f"Density: {nx.density(g_plus)}")
    # exit()

    connected_plus = check_disconnected_components(g_plus)
    connected_minus = connected_plus

    adjMat_plus = np.zeros((n_nodes, n_nodes))
    adjMat_minus = np.zeros((n_nodes, n_nodes))

    in_degree_plus = np.zeros(n_nodes)
    in_degree_minus = np.zeros(n_nodes)
    out_degree_plus = np.zeros(n_nodes)
    out_degree_minus = np.zeros(n_nodes)

    edge_counts = [0, 0]
    for [v1, v2] in edge_list:
        adjMat_plus[v1, v2] = 1
        adjMat_minus[v1, v2] = 1

        out_degree_plus[v1] += 1
        out_degree_minus[v1] += 1
        in_degree_plus[v2] += 1
        in_degree_minus[v2] += 1

        if [v2, v1] in edge_list:
            edge_counts[1] = edge_counts[1] + 1
        else:
            edge_counts[0] = edge_counts[0] + 1
    edge_counts[1] = int(edge_counts[1] / 2)
    edge_counts = np.asarray(edge_counts)
    logger.info(f"Edge Counts: {edge_counts}")

    complexity_plus = complexity(adjMat_plus, eps_bias)
    complexity_minus = complexity_plus

    logger.info("First motif count done.")

    counts_init = edge_counts.tolist()
    score_init = score(
        0,
        target_complexity,
        complexity_plus,
        p_mean,
        p_unidir,
        p_bidir,
        edge_counts,
        connected_plus,
        [1, 1, 1],
        with_density,
        with_detailed_density,
    )

    # matplotlib snapshotting init
    snp_count = 0
    counts_snp = np.asarray([counts_init], dtype=float)
    counts_snp[0] = counts_snp[0] / number_of_possible_edges

    edge_counts_rel = edge_counts / number_of_possible_edges
    connection_probability = edge_counts_rel[0] * 0.5 + edge_counts_rel[1]
    density_snp = [connection_probability]  # save average connection probability in the beginning

    # these are delacred as lists to use += in the code below in stead of np.append as that is slower
    counts_snp = counts_snp.tolist()
    score_snp = [score_init]
    # complexity_snp = [complexity]
    complexity_snp = [complexity_plus]
    step_ids = [0]
    change_tracker = np.zeros((n_iter_steps, 2), dtype=np.int8)

    time_start_iteration = t.time()

    logger.info("Starting iteration process...")

    # Glauber Dynamic iteration
    for iteration_step in range(n_iter_steps):
        logger.info(f"Starting Iteration: {iteration_step}")

        iteration_progress = iteration_step / n_iter_steps

        edge_counts_rel = edge_counts / number_of_possible_edges
        connection_probability = edge_counts_rel[0] * 0.5 + edge_counts_rel[1]
        logger.info(f"Average Connection Probability: {connection_probability}")

        removal_prob = connection_probability
        remove_edge = rng.choice(2, p=[1 - removal_prob, removal_prob])

        logger.info(f"Edge Counts: {edge_counts}")

        if remove_edge:
            logger.info("trying to remove edge")

            change_tracker[iteration_step, 0] = -1

            # Draw edge to remove
            edges = np.transpose(np.nonzero(adjMat_plus))
            edge = edges[int(rng.choice(np.arange(len(edges))))]
            inverse_exists = adjMat_plus[edge[1], edge[0]]

            # update matrices
            g_minus.remove_edge(edge[0], edge[1])
            adjMat_minus[edge[0], edge[1]] = 0
            out_degree_minus[edge[0]] -= 1
            in_degree_minus[edge[1]] -= 1

            # update connectedness
            connected_minus = check_disconnected_components(g_minus)

            # update edge counts
            edge_counts_plus = copy.deepcopy(edge_counts)
            edge_counts_minus = copy.deepcopy(edge_counts)

            if inverse_exists:
                edge_counts_minus[0] += 1
                edge_counts_minus[1] -= 1
            else:
                edge_counts_minus[0] -= 1

            # update complexity
            complexity_minus = complexity(adjMat_minus, eps_bias)
        else:
            logger.info("trying to add edge")

            change_tracker[iteration_step, 0] = 1

            # Draw edge to add
            zeros = np.argwhere(adjMat_plus == 0)
            edge = zeros[int(rng.choice(np.arange(len(zeros))))]
            inverse_exists = adjMat_plus[edge[1], edge[0]]

            # update matrices
            g_plus.add_edge(edge[0], edge[1])
            adjMat_plus[edge[0], edge[1]] = 1
            out_degree_plus[edge[0]] += 1
            in_degree_plus[edge[1]] += 1

            # update connectedness
            connected_plus = check_disconnected_components(g_plus)

            # update edge counts
            edge_counts_plus = copy.deepcopy(edge_counts)
            edge_counts_minus = copy.deepcopy(edge_counts)

            if inverse_exists:
                edge_counts_plus[0] -= 1
                edge_counts_plus[1] += 1
            else:
                edge_counts_plus[0] += 1

            # update complexity
            complexity_plus = complexity(adjMat_plus, eps_bias)

        if iteration_step % 200 == 0:
            time_iterations_current = t.time() - time_start_iteration
            time_per_iteration = time_iterations_current / (iteration_step + 1)
            logger.info("Current time per iteration: ", time_per_iteration)

        snp_count += 1

        #
        if with_detailed_density:
            edge_counts_norm = (copy.deepcopy(edge_counts) / number_of_possible_edges) / np.array([p_unidir, p_bidir])
        else:
            edge_counts_norm = (
                (edge_counts / number_of_possible_edges)[0] * 0.5 + (edge_counts / number_of_possible_edges)[1]
            ) / p_mean

        if remove_edge:
            complexity_norm = complexity_plus / target_complexity
        else:
            complexity_norm = complexity_minus / target_complexity

        weights = set_weights(complexity_norm, edge_counts_norm, with_detailed_density)
        logger.info(f"Weights: {weights}")

        # calculate difference between f(M+) and f(M-), save, and update if we transition
        score_g_plus = score_diff_inflation * score(
            iteration_progress,
            target_complexity,
            complexity_plus,
            p_mean,
            p_unidir,
            p_bidir,
            edge_counts_plus / number_of_possible_edges,
            connected_plus,
            weights,
            with_density,
            with_detailed_density,
        )

        score_g_minus = score_diff_inflation * score(
            iteration_progress,
            target_complexity,
            complexity_minus,
            p_mean,
            p_unidir,
            p_bidir,
            edge_counts_minus / number_of_possible_edges,
            connected_minus,
            weights,
            with_density,
            with_detailed_density,
        )

        logger.info(f"Score Plus: {score_g_plus}")
        logger.info(f"Score Minus: {score_g_minus}")

        # calculate proability to remove chosen edge (adding it is 1 - prob)
        prob_minus = 1 / (1 + np.exp(score_g_plus - score_g_minus))
        logger.info(f"Prob Minus: {prob_minus}")

        # chose whether to remove edge or not
        choice = rng.choice(2, p=[prob_minus, 1 - prob_minus])

        #
        if choice == 0:
            if remove_edge:
                edge_counts = copy.deepcopy(edge_counts_minus)
                change_tracker[iteration_step, 1] = 1
            else:
                change_tracker[iteration_step, 1] = 0

            g_plus.remove_edge(edge[0], edge[1])
            adjMat_plus[edge[0], edge[1]] = 0
            out_degree_plus[edge[0]] -= 1
            in_degree_plus[edge[1]] -= 1
            complexity_plus = complexity_minus
            connected_plus = connected_minus

        #
        if choice == 1:
            if remove_edge:
                change_tracker[iteration_step, 1] = 0
            else:
                edge_counts = copy.deepcopy(edge_counts_plus)
                change_tracker[iteration_step, 1] = 1

            g_minus.add_edge(edge[0], edge[1])
            adjMat_minus[edge[0], edge[1]] = 1
            out_degree_minus[edge[0]] += 1
            in_degree_minus[edge[1]] += 1
            complexity_minus = complexity_plus
            connected_minus = connected_plus

        logger.info(f"Target complexity: {target_complexity}")
        logger.info(f"Current complexity: {complexity_plus}")

        if snp_count == sampling_interval:
            density_snp.append(connection_probability)  # save average connection probability

        if choice == 0 and snp_count == sampling_interval:
            counts_snp += [edge_counts_minus / number_of_possible_edges]
            score_snp += [score_g_minus / get_max_score(score_diff_inflation, weights)]
            complexity_snp += [complexity_minus]
            step_ids += [iteration_step]
            snp_count = 0
        elif choice == 1 and snp_count == sampling_interval:
            counts_snp += [edge_counts_plus / number_of_possible_edges]
            score_snp += [score_g_plus / get_max_score(score_diff_inflation, weights)]
            complexity_snp += [complexity_plus]
            step_ids += [iteration_step]
            snp_count = 0

        if choice == 0 and score_g_minus >= score_convergence_cutoff * get_max_score(score_diff_inflation, weights):
            n_iter_steps = iteration_step
            counts_snp += [edge_counts_minus / number_of_possible_edges]
            score_snp += [score_g_minus / get_max_score(score_diff_inflation, weights)]
            complexity_snp += [complexity_minus]
            step_ids += [iteration_step]
            density_snp.append(connection_probability)  # save average connection probability
            logger.info(f"Breaking loop at iteartion {iteration_step}")
            logger.info(score_g_minus)
            logger.info(score_convergence_cutoff * get_max_score(score_diff_inflation, weights))
            logger.info(get_max_score(score_diff_inflation, weights))
            break
        elif choice == 1 and score_g_plus >= score_convergence_cutoff * get_max_score(score_diff_inflation, weights):
            n_iter_steps = iteration_step
            counts_snp += [edge_counts_plus / number_of_possible_edges]
            score_snp += [score_g_plus / get_max_score(score_diff_inflation, weights)]
            complexity_snp += [complexity_plus]
            step_ids += [iteration_step]
            density_snp.append(connection_probability)  # save average connection probability
            logger.info(f"Breaking loop at iteartion {iteration_step}")
            logger.info(score_g_plus)
            logger.info(score_convergence_cutoff * get_max_score(score_diff_inflation, weights))
            logger.info(get_max_score(score_diff_inflation, weights))
            break

    time_finish_iteration = t.time()

    # calculate runtime in seconds
    time_finish_total = t.time()
    time_iterations = time_finish_iteration - time_start_iteration
    time_total = time_finish_total - time_start_total

    counts_snp = np.asarray(counts_snp)
    score_snp = np.asarray(score_snp)

    return (
        adjMat_plus,
        counts_snp,
        complexity_snp,
        density_snp,
        score_snp,
        step_ids,
        change_tracker,
        time_total,
        time_iterations,
    )


def complexity(adjMat, eps_bias):
    # return np.log(np.max(np.abs(np.linalg.eigvalsh(adjMat))))
    x = np.real(np.max(np.abs(np.linalg.eigvals(adjMat))))
    return np.log(max(x, eps_bias))


def set_weights(complexity, edges, with_detailed_density):
    if with_detailed_density:
        arr = np.array([complexity, edges[0], edges[1]])
    else:
        arr = np.array([complexity, edges])
    concentration_difference = np.where(arr == 0, 0.000001, arr)
    out = np.where(concentration_difference <= 1, 1 / concentration_difference, concentration_difference)
    out = 2 * (3 * (out - 1)) ** 2 + 1
    # out[0] = out[0] * sum(out[1:])
    return out


def get_max_score(inflation, weights):
    return inflation * sum(weights)


def score(
    iteration_progress,
    target_complexity,
    curr_complexity,
    average_connectivity,
    uni_conn,
    bi_conn,
    edges,
    conn_penalty,
    weights,
    with_density=False,
    with_detailed_density=False,
):
    score = 0
    if with_detailed_density:
        score = hamiltonian_dirac(
            iteration_progress, target_complexity, curr_complexity, [uni_conn, bi_conn], edges, weights, conn_penalty
        )
    elif with_density:
        score = hamiltonian_dirac(
            iteration_progress,
            target_complexity,
            curr_complexity,
            [average_connectivity],
            [edges[0] * 0.5 + edges[1]],
            weights,
            conn_penalty,
        )
    else:
        score = hamiltonian_dirac(
            iteration_progress, target_complexity, curr_complexity, weights=weights, connectivity_penalty=conn_penalty
        )
    return score


def hamiltonian_dirac(
    iteration_progress,
    target_complexity,
    curr_complexity,
    target_edges=None,
    curr_edges=None,
    weights=None,
    connectivity_penalty=None,
):
    n_p = 1
    edge_score = 0
    complexity_score = 0

    if target_edges:
        if len(target_edges) == 1:
            edge_score = (
                1
                / dirac_delta_approx(0, 0.3, iteration_progress)
                * (weights[1] * dirac_delta_approx(curr_edges[0] / target_edges[0] - 1, 0.3, iteration_progress))
            )
            n_p = 2
        else:
            edge_score = (
                1
                / dirac_delta_approx(0, 0.3, iteration_progress)
                * (
                    weights[1] * dirac_delta_approx(curr_edges[0] / target_edges[0] - 1, 0.3, iteration_progress)
                    + weights[2] * dirac_delta_approx(curr_edges[1] / target_edges[1] - 1, 0.3, iteration_progress)
                )
            )

            n_p = 3

    complexity_score = (
        1
        / dirac_delta_approx(0, 0.3, iteration_progress)
        * weights[0]
        * dirac_delta_approx(curr_complexity / target_complexity - 1, 0.3, iteration_progress)
    )

    return (
        edge_score + complexity_score - n_p * connectivity_penalty
    )  # 15 because thats the number of motifs considered rn


# If a graph is not connected we add a big penalty, that is even bigger, the more disconnected the graph is
def check_disconnected_components(g: nx.DiGraph):
    number_of_components = nx.number_weakly_connected_components(g)
    return number_of_components - 1


# Dirac + quadratic function with max value at x = 0 (normalized to 1 in parent function call), fall of region left and right of the center can be govened by weight
def dirac_delta_approx(x, weight, iteration_progress):
    # return 1/(weight * np.sqrt(np.pi)) * np.exp(-((x)/weight)**2) #pure
    if iteration_progress < 0.8:  # switching
        return -(x**2) + 1
    else:
        return 1 / (weight * np.sqrt(np.pi)) * np.exp(-(((x) / weight) ** 2)) + (-(x**2) + 1)  # mixed
