# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import get_type_hints, get_origin

# internal imports
from symseq.utils.strtools import string_as_symbols

# from symseq.metrics import complexity
from symseq import metrics


# @staticmethod
def process_feature(string: list[str], train_set: list[list[str]], features: dict) -> dict:
    """
    For a given string, compute the specified (complexity or statistical) features and return them as a dictionary. The
    user is responsible for passing correct function arguments to the relevant metrics.

    Parameters
    ----------
    string : list of str
        The string to compute features for.
    train_set : list of list of str
        The training set to use for computing features.
    features : dict
        A dictionary of features to compute. Keys are feature names, values are dictionaries with additional
        parameters for the feature computation.

    Returns
    -------
    results : dict
        A dictionary with feature names as keys and feature values as values.
    """
    results = {}

    for feature_name, feature_dict in features.items():
        if "kwargs" not in feature_dict:
            feature_dict["kwargs"] = {}  # default kwargs

        if feature_name not in metrics.__all__:
            raise ValueError(f"Unknown feature {feature_name}. Available features/metrics: {metrics.__all__}")

        func = getattr(metrics, feature_name)
        if list(get_type_hints(func).values())[1] == list[str]:
            res = func(train_set, string, **feature_dict["kwargs"])
        elif list(get_type_hints(func).values())[1] == list[list[str]]:
            res = func(train_set, [string], **feature_dict["kwargs"])
        else:
            raise ValueError(f"Unexpected function argument type for feature {feature_name}")

        if isinstance(res, list):
            assert len(res) == 1, f"Feature {feature_name} returned a list of length > 1"
            results[feature_name] = res[0]
        else:
            assert isinstance(res, float), f"Feature {feature_name} returned a non-float value"
            results[feature_name] = res

    return results


# @staticmethod
def dfs_from_partial(G, path, kwargs):
    """Continue DFS from a given partial path."""
    # results_set = set()
    strings = []

    def dfs(p):
        last = p[-1]
        # is_terminal = G.nodes[last].state.terminal
        is_terminal = kwargs["eos"] == last
        # Only collect if we're at a terminal node
        if len(p) > 1 and is_terminal:
            if len(p) - 1 < kwargs["length_range"][0]:  # check for min length
                return

            string = string_as_symbols(p[:-1])  # remove EOS
            strings.append(tuple(string))
            # results_set.add("".join(string))
        if len(p) - 1 == kwargs["length_range"][1]:  # discard paths that are too long
            return
        for nbr in G.neighbors(last):
            dfs(p + [nbr])

    dfs(path)
    return strings


# @staticmethod
def all_paths_as_strings(G, start_nodes, workers=4, **kwargs):
    """
    TODO double-check
    """
    assert "length_range" in kwargs
    assert "eos" in kwargs
    # Heuristic to calculate the depth of the split."""
    frontier = [[s] for s in start_nodes]
    split_depth = 0
    max_depth = kwargs["length_range"][1]
    min_tasks_factor = 2
    while len(frontier) < workers * min_tasks_factor and split_depth < max_depth:
        new_frontier = []
        for path in frontier:
            last = path[-1]
            for nbr in G.neighbors(last):
                new_frontier.append(path + [nbr])
        frontier = new_frontier
        split_depth += 1

    # Step 2: assign each frontier path to a worker
    all_results = set()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(dfs_from_partial, G, path, kwargs) for path in frontier]
        for f in as_completed(futures):
            string_tuples = f.result()
            all_results |= set(string_tuples)

    return all_results


###################################################################################################################
# Prediction-related functions which could be moved to a separate module
###################################################################################################################

# def compute_true_prediction_probabilities_graph(self, max_n_step, max_str_len, verbose=False):
#     """
#         Computes the true probabilities for the specified maximum n-step predictions.
#     # def compute_true_prediction_probabilities_graph(self, max_n_step, max_str_len, verbose=False):
#     #"""

#     Computes the true probabilities for the specified maximum n-step predictions.

#     Parameters
#     ----------
#     max_n_step

#     Notes
#     -----
#     This function currently assumes that the all initial states have equal starting probability.

#     Returns
#     -------

#     """
#     logger.info(f"Computing true probabilities for a maximum of {max_n_step}-step prediction...")
#     P_computed = defaultdict(dict)  # for each context, it stores
#     P_computed_normalized = defaultdict(dict)
#     P_tokens = defaultdict(dict)
#     prob_context = {}  # probability of nonindexed context (string prefixes) - sum over all indexed path probs

#     # build graph corresponding to grammar
#     if self.graph is None:
#         self.graph = self.create_graph()

#     p_start_state = 1.0 / len(self.start_states)
#     for start_state in self.start_states:
#         # Start DFS traversal from the start_node
#         self._contextual_predictions_dfs(
#             # start_state, [start_state], 0, max_n_step, p_start_state, P_computed, prob_context
#             start_state,
#             [start_state],
#             0,
#             max_str_len,
#             p_start_state,
#             P_computed,
#             prob_context,
#         )
#     # for each context, compute the all the predictable substrings along with the correct, final probability
#     for context_string, predictions_dict in sorted(P_computed.items()):
#         for i in range(1, max_n_step + 1):
#             for prediction_string, probabilities in predictions_dict.items():
#                 if len(prediction_string) == i:
#                     P_computed_normalized[context_string][prediction_string] = (
#                         sum(probabilities) / prob_context[context_string]
#                     )
#     # for each context and for each n-step, reduce the predicted substrings to its last token - we are interested
#     # only in the probability of the predicted token given a certain context.
#     for context_string, predictions_dict in sorted(P_computed_normalized.items()):
#         for i in range(1, max_n_step + 1):
#             for predicted_substring, p in predictions_dict.items():
#                 if len(predicted_substring) == i:
#                     last_token = predicted_substring[-1]
#                     try:
#                         P_tokens[context_string][i][last_token] += p
#                     except:
#                         P_tokens[context_string][i] = defaultdict(float)
#                         P_tokens[context_string][i][last_token] = p
#     if verbose:
#         logger.info("Computed probabilities, normalized")
#         pprint.pprint(P_computed_normalized)
#     self.prob_substring_predictions = P_computed_normalized
#     self.prob_token_predictions = P_tokens  # only single tokens, summed over all possible substrings ending with it

# def compute_true_probabilities_sampling(self, max_str_len, max_n_step, n_samples=int(1e6), verbose=False):
#     """
#     Compute the true probabilities for n-step prediction by sampling from the grammar and calculating the
#     frequencies manually.

#     Parameters
#     ----------
#     max_str_len
#     max_n_step
#     n_samples
#     verbose

#     Returns
#     -------

#     """
#     logger.info("Calculating true transition probabilities using the sampling approach...")
#     cache_string_sets = self.cache_string_set()  # save here because it will be overwritten temporarily
#     self.generate_string_set(
#         n_strings=n_samples,
#         length_range=(1, 2 * max_str_len),  # enough to sample
#         soft_lower_bound=True,
#         remove_eos=True,
#         only_grammatical=True,
#     )

#     P_sampled = defaultdict(dict)
#     P_tokens = defaultdict(dict)
#     string_set = ["".join(x) + "#" for x in self.string_set]
#     unique_strings = np.unique(string_set).tolist()
#     all_prefixes = [
#         prefix for string in string_set for prefix in self.get_all_prefixes(string) if len(prefix) <= max_str_len
#     ]
#     prefix_frequency = Counter(all_prefixes)

#     for string in unique_strings:
#         string = string[:max_str_len]
#         for context_pos in range(1, len(string)):
#             for pred_pos in range(context_pos + 1, len(string) + 1):
#                 context_string = string[:context_pos]
#                 pred_string = string[context_pos:pred_pos]
#                 concatenated_string = context_string + pred_string
#                 P_sampled[context_string][pred_string] = (
#                     prefix_frequency[concatenated_string] / prefix_frequency[context_string]
#                 )

#     for context_string, predictions_dict in sorted(P_sampled.items()):
#         for i in range(1, max_n_step + 1):
#             for predicted_substring, p in predictions_dict.items():
#                 if len(predicted_substring) == i:
#                     last_token = predicted_substring[-1]
#                     try:
#                         P_tokens[context_string][i][last_token] += p
#                     except:
#                         P_tokens[context_string][i] = defaultdict(float)
#                         P_tokens[context_string][i][last_token] = p
#     if verbose:
#         logger.info("Sampled probabilities")
#         logger.info(pprint.pformat(P_sampled))
#     self.restore_string_set(**cache_string_sets)  # revert to previous string set
#     self.prob_token_predictions = P_tokens
#     return P_sampled

# def _prediction_subgraph_dfs(self, state, depth, depth_limit, predicted_states, P_subgraph, path_prob):
#     """
#     Recursively computes the prediction subgraph and the probability of generating all possible substrings
#     that can follow a given state in a directed graph (subgraph). This is done via depth-first search (DFS),
#     exploring the graph up to a specified depth limit and calculating the total probability for each path.

#     For each state reached during the DFS, the method concatenates the predicted states (without indices) into a
#     string and stores the cumulative probability of generating that substring in the `P_subgraph` dictionary.

#     Parameters
#     ----------
#     state : str
#         The current state or node in the graph from which to explore.
#     depth : int
#         The current depth in the DFS recursion, indicating how far the search has progressed from the initial state.
#     depth_limit : int
#         The maximum depth allowed for the DFS exploration. The recursion will stop once this depth is reached.
#     predicted_states : list[str]
#         A list of predicted states visited so far (with indices included) in the current DFS path.
#         Each state contributes to forming a predicted substring.
#     P_subgraph : dict
#         A dictionary to store the probabilities of each predicted substring. The keys are the predicted substrings
#         (concatenated states without indices), and the values are the cumulative probabilities for those substrings.
#     path_prob : float
#         The cumulative probability of the current DFS path. This is updated as the DFS progresses through the graph,
#         with the default starting value of 1.0.

#     Returns
#     -------
#     None
#         The function updates `P_subgraph` in-place and does not return any values.

#     Notes
#     -----
#     - This function performs a DFS on a directed graph `G`, starting from the specified `state`.
#     - For each visited state, it records the corresponding substring formed by concatenating the first character of
#       each state in `predicted_states`.
#     - The cumulative probability of reaching a state is calculated as the product of transition probabilities along the
#       path from the initial state.
#     - Recursion terminates when the specified depth limit is reached.
#     - If a predicted substring already exists in `P_subgraph`, its cumulative probability is updated by adding the
#       current path probability; otherwise, a new entry is created.
#     - The function can be useful in scenarios involving probabilistic state transitions, such as Markov models or
#       predictive text generation.
#     """
#     # if depth > depth_limit or state == self.eos:
#     if depth > depth_limit:
#         return

#     predicted_tokens = [s[0] for s in predicted_states]  # predicted states without indices
#     pred_tokens_string = "".join(predicted_tokens)  # concatenated as a string

#     if pred_tokens_string in P_subgraph:
#         P_subgraph[pred_tokens_string] += path_prob
#     else:
#         P_subgraph[pred_tokens_string] = path_prob

#     # print(
#     #     f"Added prediction substring {predicted_states} ({predicted_tokens}) with {path_prob}. "
#     #     f"Total probability: {P_subgraph[pred_tokens_string]}"
#     # )

#     # Explore neighbors (outgoing edges)
#     for neighbor in self.graph.successors(state):
#         edge_data = self.graph.get_edge_data(state, neighbor)  # edge data (which includes the weight)
#         prob = edge_data["weight"]

#         self._prediction_subgraph_dfs(
#             state=neighbor,
#             depth=depth + 1,
#             depth_limit=depth_limit,
#             predicted_states=predicted_states + [neighbor],
#             P_subgraph=P_subgraph,
#             path_prob=path_prob * prob,
#         )

# def _contextual_predictions_dfs(
#     self, state, context_states, depth, depth_limit, path_prob, P_contextual_predictions, prob_context
# ):
#     """
#     Recursively computes the context-aware prediction subgraph using depth-first search (DFS) and updates the
#     probabilities of all possible substrings that can follow a sequence of states (context) in a directed graph.

#     This function performs two tasks:
#     1. Computes the prediction subgraph from a given state using `dfs_prediction_subgraph`, which tracks substrings
#        and their probabilities from neighboring states.
#     2. Recursively explores the graph by adding the current state to the context and accumulating path
#        probabilities. The computed subgraphs and their probabilities are stored in `P_contextual_predictions`.

#     Parameters
#     ----------
#     state : str
#         The current state or node from which the DFS exploration begins.
#     context_states : list[str]
#         A list of states (with indices) representing the context (the sequence of states visited so far). This context
#         is used to generate context-based substrings.
#     depth : int
#         The current depth of the DFS recursion, indicating the level of exploration from the initial state.
#     depth_limit : int
#         The maximum depth to explore in the DFS. Once this limit is reached, the recursion stops.
#     path_prob : float
#         The cumulative probability of the current path through the graph, computed by multiplying the transition
#         probabilities along the path.
#     P_contextual_predictions : dict
#         A dictionary to store the computed probabilities of prediction substrings. The keys are context substrings
#         (concatenated state names without indices), and the values are dictionaries mapping predicted substrings
#         to lists of cumulative probabilities.
#     prob_context : dict
#         A dictionary to store the cumulative probability of each context substring (without indices). The keys are
#         context substrings, and the values are their cumulative probabilities along explored paths.

#     Returns
#     -------
#     None
#         This function updates `P_computed` and `prob_context` in-place and does not return any values.

#     Notes
#     -----
#     - The function performs DFS starting from the given `state` and explores neighboring states (successors).
#     - For each state, it computes the prediction subgraph using `dfs_prediction_subgraph`, which calculates the
#       probabilities of substrings starting from that state.
#     - As the DFS progresses, the context is expanded by adding the current state to `context_states`, and the
#       cumulative path probability is updated by multiplying the transition probabilities.
#     - The results (prediction substrings and their probabilities) are stored in `P_computed`, a nested dictionary
#       where keys represent context substrings, and values are lists of probabilities for corresponding predicted
#       substrings.
#     - If a context substring already exists in `P_contextual_predictions`, the new probabilities are added to the
#       existing ones; otherwise, a new entry is created.
#     - The function can be used to model state-based prediction systems, such as those found in Markov chains or
#       probabilistic state transition models.
#     """
#     # if depth > depth_limit or state == self.eos:
#     if depth > depth_limit:
#         return

#     # Explore neighbors (outgoing edges)
#     P_subgraph = dict()
#     for neighbor in self.graph.successors(state):
#         edge_data = self.graph.get_edge_data(state, neighbor)  # edge data (which includes the weight)
#         prob_neighbor = edge_data["weight"]
#         # compute prediction subgraph starting at the selected neighbor
#         self._prediction_subgraph_dfs(
#             state=neighbor,
#             depth=depth + 1,
#             predicted_states=[neighbor],
#             path_prob=prob_neighbor,
#             depth_limit=depth_limit,
#             P_subgraph=P_subgraph,
#         )
#         # recursive call by increasing context
#         self._contextual_predictions_dfs(
#             neighbor,
#             context_states + [neighbor],
#             depth + 1,
#             depth_limit,
#             path_prob * prob_neighbor,
#             P_contextual_predictions,
#             prob_context,
#         )

#     context_tokens = [s[0] for s in context_states]  # context states without indices
#     context_tokens_string = "".join(context_tokens)  # concatenated as string

#     if context_tokens_string in P_contextual_predictions:
#         prob_context[context_tokens_string] += path_prob
#         for pred_tokens_string, p in P_subgraph.items():
#             try:
#                 P_contextual_predictions[context_tokens_string][pred_tokens_string] += [path_prob * p]
#             except:
#                 P_contextual_predictions[context_tokens_string][pred_tokens_string] = [path_prob * p]
#     else:
#         prob_context[context_tokens_string] = path_prob
#         for pred_tokens_string, p in P_subgraph.items():
#             P_contextual_predictions[context_tokens_string][pred_tokens_string] = [path_prob * p]

#     # logger.info(
#     #     f"\n\tCurrent state: {state}"
#     #     f"\n\tContext states: {context_states}"
#     #     f"\n\tContext tokens: {context_tokens}"
#     #     f"\n\tContext tokens string: {context_tokens_string}"
#     #     f"\n\tGenerated subgraph from current state {state}: {P_subgraph}"
#     #     f"\n\tUpdated global graph P to: {pprint.pformat(P_contextual_predictions, indent=2)}"
#     # )

# def create_graph(self):
#     graph = nx.DiGraph()
#     for transition in self.transitions:
#         source, target, probability = transition
#         graph.add_edge(source, target, weight=probability)
#     return graph
