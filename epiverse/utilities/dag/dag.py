import networkx as nx
import numpy as np

from typing import Dict, Tuple, List, Callable, Set
from inspect import signature
from scipy.special import logit, expit


class DAG:
    def __init__(self, node_dict: Dict, edge_list: List, n: int = 1000, **kwargs):
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

        if n is not None:
            if n <= 0:
                raise Exception(
                    f"Provided N must be greater than 0, but is {n}")
            self.n = n

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

    def _check_proportion(proportion: float) -> None:
        if proportion < 0 or proportion > 1:
            raise Exception(
                f"Proportion provided must be in range [0,1], was: {proportion}")

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

    def get_node_function(self, node: str) -> Callable:
        if node not in self.node_dict.keys():
            raise Exception(
                f"Node {node} not present in DAG that contains {self.node_dict.keys()}")

        return self.node_dict[node]

    def get_ancestors_from_set(self, set_of_ancestors: Set) -> List[List]:
        if not set_of_ancestors:  # Set is empty
            return [[]]

    def binary_confounding_triangle(mean_a: float, mean_l: float, mean_y0: float, mean_y1: float, n: int):
        DAG._check_proportion(mean_a)
        DAG._check_proportion(mean_l)
        DAG._check_proportion(mean_y0)
        DAG._check_proportion(mean_y1)

        if n is not None and n <= 0:
            raise Exception(f"Provided N must be greater than 0, but is {n}")
        # Situations where using default - override before.
        if n is None and self.n is not None:
            n = self.n

        bct_edge_list = [
            ("A", "Y"),
            ("L", "Y"),
            ("L", "A")
        ]

        rng = np.random.default_rng()

        def L_func():
            return rng.binomial(1, mean_l, n)

        def A_func(L):
            # TODO: change strength of difference.
            logit_proportion = logit(mean_a) - mean_l + L
            return rng.binomial(1, expit(logit_proportion), n)

        def Y_func(A, L):
            y0_l = np.sum((1-A) * L) / np.sum(1-A)
            y1_l = np.sum(A * L) / np.sum(A)
            y0_logit_proportion = (1-A) * (logit(mean_y0) - y0_l + L)
            y1_logit_proportion = A * (logit(mean_y1) - y1_l + L)

            return rng.binomial(1, expit(y0_logit_proportion + y1_logit_proportion), n)

        bct_node_dict = {
            "L": L_func,
            "A": A_func,
            "Y": Y_func
        }

        return DAG(bct_node_dict, bct_edge_list)

    def binary_m_bias(mean_a: float, mean_m: float, mean_y0: float, mean_y1: float, mean_u1: float, mean_u2: float, n: int):
        DAG._check_proportion(mean_a)
        DAG._check_proportion(mean_m)
        DAG._check_proportion(mean_y0)
        DAG._check_proportion(mean_y1)
        DAG._check_proportion(mean_u1)
        DAG._check_proportion(mean_u2)

        if n is not None and n <= 0:
            raise Exception(f"Provided N must be greater than 0, but is {n}")
        # Situations where using default - override before.
        if n is None and self.n is not None:
            n = self.n

        mbias_edge_list = [
            ("A", "Y"),
            ("U1", "A"),
            ("U1", "M"),
            ("U2", "Y"),
            ("U2", "M"),
        ]

        rng = np.random.default_rng()

        def u1_func():
            return rng.binomial(1, mean_u1, n)

        def u2_func():
            return rng.binomial(1, mean_u2, n)

        def A_func(U1):
            logit_proportion = logit(mean_a) - mean_u1 + U1
            return rng.binomial(1, expit(logit_proportion), n)

        def Y_func(A, U2):
            y0_l = np.sum((1-A) * U2) / np.sum(1-A)
            y1_l = np.sum(A * U2) / np.sum(A)
            y0_logit_proportion = (1-A) * (logit(mean_y0) - y0_l + U2)
            y1_logit_proportion = A * (logit(mean_y1) - y1_l + U2)

            return rng.binomial(1, expit(y0_logit_proportion + y1_logit_proportion), n)

        def M_func(U1, U2):
            m_logit_proportion = logit(
                mean_m) - (mean_u1 - U1) - (mean_u2 - U2)

            return rng.binomial(1, expit(m_logit_proportion), n)

        mbias_node_dict = {
            "A": A_func,
            "M": M_func,
            "Y": Y_func,
            "U1": u1_func,
            "U2": u2_func
        }

        return DAG(mbias_node_dict, mbias_edge_list)
