import networkx as nx
import pandas as pd
from typing import List, Tuple

from epiverse.utilities.data_generation.data_generator import DataGenerator
from epiverse.utilities.dag.dag import DAG


class DataGeneratorDAG(DataGenerator):

    def __init__(self, dag: DAG):
        self.dag = dag

    def generate_data(self):

        if self.dag is None:
            raise Exception("Provided DAG is not a DAG.")

        generations = nx.topological_generations(self.dag.graph_structure)

        data_dictionary = {}
        for gen in generations:
            for node in gen:
                # Get all ancestors (can be empty) and apply to function
                ancestors = self.dag.graph_structure.predecessors(node)
                if not ancestors:
                    data_dictionary[node] = self.dag.get_node_function(node)()
                    continue

                ancestor_data = [data_dictionary[anc] for anc in ancestors]
                data_dictionary[node] = self.dag.get_node_function(
                    node)(*ancestor_data)

        return pd.DataFrame(data_dictionary)
