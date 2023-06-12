import networkx as nx
from typing import List, Tuple

from epiverse.utilities.data_generation.data_generator import DataGenerator
from epiverse.utilities.dag.dag import DAG


class DataGeneratorDAG(DataGenerator):

    def __init__(self, dag: DAG):
        self.dag = dag

    def generate_data(self):

        if self.dag is None:
            raise Exception("Provided DAG is not a DAG.")

        generations = nx.topological_generations(self.dag)

        for gen in generations:
            print(gen)
