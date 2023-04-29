from src.utilities.data_generation.data_generator import DataGenerator
from typing import List
import pandas as pd
from shapely.geometry import Point, Polygon


class DataGeneratorPolygon(DataGenerator):

    def __init__(self, *args, labels: List):
        # From input list of 2D points, make triangulations
        pass

    def generate_data(self, n: int = 1000) -> pd.DataFrame:
        pass

    def construct_polygon(self, *args) -> Polygon:
        pass
