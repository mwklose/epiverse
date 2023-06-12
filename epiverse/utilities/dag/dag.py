import networkx as nx
import numpy as np

from typing import Dict, Tuple, List, Callable
from inspect import signature
from scipy.special import logit, expit


class DAG:
    def __init__(self, node_dict: Dict, edge_list: List, **kwargs):
        """Initializes a DAG based on Node labels, Node functions, and Edges. 

        Args:
            node_dict (Dict): A dictionary corresponding to the nodes and their corresponding data generating functions. 
            edge_list (List): A list with all edges to add, which is passed onto NetworkX. 
        """
        self.graph_structure = None
        self.node_dict, self.edge_list = self._check_nodes_and_edges(
            node_dict, edge_list)

        self.graph_structure = self._check_graph_structure(
            self.node_dict, self.edge_list)

        self.params = kwargs
        self._rng = np.random.default_rng()
        self.n = 0

    def _check_nodes_and_edges(self, node_dict, edge_list) -> Tuple[Dict, Dict]:
        # Edges:
        # Create Set of Edges to eventually see if all Nodes are present.
        # Check for acyclic is done later on.
        source_edges = set([u for u, _ in edge_list])
        target_edges = set([v for _, v in edge_list])
        edge_set = source_edges.union(target_edges)

        # Nodes: either initializing or adding on top of present DAG.
        if self.graph_structure is None:
            provided_node_set = set(node_dict.keys())
        else:
            present_nodes = set(self.node_dict.keys())
            provided_node_set = set(node_dict.keys()).union(present_nodes)

        node_set_difference = edge_set.difference(provided_node_set)

        if len(node_set_difference) != 0:
            raise Exception(
                f"Nodes {node_set_difference} were not provided function definitions.")

        # Nodes:
        # Check that each node function has the correct number of arguments
        # To start, need to find how many arguments are needed.
        node_indegree = {v: []
                         for v in edge_set}

        for u, v in edge_list:
            node_indegree[v].append(u)

        # Next, iterate through each node function and see how many arguments there are.
        for node_k, node_v in node_dict.items():
            number_of_arguments = len(signature(node_v).parameters)

            if len(node_indegree[node_k]) != number_of_arguments:
                raise Exception(
                    f"Node {node_k} function takes in {number_of_arguments} arguments, but receives {len(node_indegree[node_k])} arguments")

        return node_dict, edge_list

    def _check_graph_structure(self, node_dict: Dict, edge_dict: Dict) -> nx.DiGraph:
        dag = nx.DiGraph(edge_dict)
        # Need to check for Cycles, but that is basically it.
        if not nx.is_directed_acyclic_graph(dag):
            raise Exception(f"Provided DAG {dag} contains cycles.")

        return dag

    def _add_nodes(self, node_dict: Dict):
        if node_dict is None:
            return

        node_keys = node_dict.keys()
        self.graph_structure.add_nodes_from(node_keys)

        self.node_dict = self.node_dict | node_dict

        return

    def _add_edges(self, edge_list: List):
        if edge_list is None:
            return

        self.graph_structure.add_edges_from(edge_list)
        return

    def add_nodes_edges(self, node_dict: Dict, edge_list: List):
        # Could add nodes or edges, or either independently.
        # Handle nodes first, then edges, then check.
        self._add_nodes(node_dict)
        self._add_edges(edge_list)
        self.node_dict, self.edge_list = self._check_nodes_and_edges(
            node_dict, edge_list)

    def binomial_one_variable(mean: float, odds_ratio: float) -> Callable:

        if mean < 0 or mean > 1:
            raise Exception(f"Mean must be in range [0, 1], was: {mean}")
        if odds_ratio <= 0:
            raise Exception(
                f"Odds ratio must be on interval (0, ∞), was: {odds_ratio}")

        return lambda x: expit(logit(mean) - np.log(odds_ratio) * (np.mean(x) - x))

    def binary_confounding_triangle(mean_a: float, mean_l: float, mean_y0: float, mean_y1):
        DAG._check_proportion(mean_a)
        DAG._check_proportion(mean_l)
        DAG._check_proportion(mean_y0)
        DAG._check_proportion(mean_y1)

        # TODO: need to find better flow to generate random values.
        bct_edge_list = [
            ("A", "Y"),
            ("L", "Y"),
            ("L", "A")
        ]

        bct_node_dict = {
            "L": lambda: self._rng.binomial(1, mean_l, self.n),
            "A": lambda L: self._rng.binomial(1, expit(logit(mean_a) - (mean_l + L)), self.n),
            "Y": lambda A, L: self._rng.binomial(1, expit(
                (1-A) * (logit(mean_y0) - (mean_l - L)) +
                A * (logit(mean_y1) - (mean_L - L))), self.n)
        }

        return DAG(bct_node_dict, bct_edge_list)

    def _check_proportion(proportion: float) -> None:
        if proportion < 0 or proportion > 1:
            raise Exception(
                f"Proportion provided must be in range [0,1], was: {proportion}")
