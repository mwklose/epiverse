import networkx as nx
import unittest


from epiverse.utilities.dag.dag import DAG


class TestDAG(unittest.TestCase):

    def test_graph_initialization(self):
        empty_dag = DAG(node_dict={}, edge_list=[])

        self.assertEqual(empty_dag.graph_structure.nodes, nx.DiGraph().nodes)
        self.assertEqual(empty_dag.graph_structure.edges, nx.DiGraph().edges)

        with self.assertRaises(Exception, msg="Node Y function takes in 1 arguments, but receives 2 arguments"):
            empty_dag.add_nodes_edges(
                node_dict={"A": lambda: 1,
                           "L": lambda: 0,
                           "Y": lambda x: x + 2},
                edge_list=[("A", "Y"), ("L", "Y")]
            )

        empty_dag.add_nodes_edges(
            node_dict={"A": lambda: 1,
                       "L": lambda: 0,
                       "Y": lambda x, y: x + y + 2},
            edge_list=[("A", "Y"), ("L", "Y")]
        )

        self.assertEqual(
            set(empty_dag.graph_structure.nodes),
            set(["A", "L", "Y"])
        )

        empty_dag.add_nodes_edges({"W": DAG.binomial_one_variable(0.5, 1.5)},
                                  edge_list=[("L", "W")])

        self.assertEqual(
            set(empty_dag.graph_structure.nodes),
            set(["A", "L", "Y", "W"])
        )
