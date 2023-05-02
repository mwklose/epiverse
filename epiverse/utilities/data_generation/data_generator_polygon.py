from epiverse.utilities.data_generation.data_generator import DataGenerator
from epiverse.utilities.geometry.polygon import EpiPolygon

from typing import List
import pandas as pd
import numpy as np


class DataGeneratorPolygon(DataGenerator):

    def __init__(self, labels: List, *args):
        self.labels = labels
        self.polygon = self.construct_polygon(args)

    def generate_data(self, n: int, seed: int = 0) -> pd.DataFrame:
        # For triangulations in a polygon, probability of selection is equal to the contribution to the total area.
        # So, we can sample from all triangles, then generate affine combination for a random point.
        if not seed:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(seed=seed)

        random_samples = rng.choice(
            self.polygon.triangulation, size=n, replace=True, p=self.polygon.triangulation_probability)

        # Select 2 uniform
        # Then multiply difference between vector and origin
        # by those 2 uniform RVs. If the sum of RVs > 1,
        # Then do a reflection.
        # Finally, add to the original point

        affine_long = rng.uniform(size=2 * n)
        affine = np.reshape(affine_long, (-1, 2))
        row_sums = affine.sum(axis=1)

        # This performs a 180 degree reflection to maintain uniformity.
        # Normalizing 3 uniform RVs does not guarantee uniformity.
        affine = (row_sums.reshape((-1, 1)) > 1) * \
            (1-affine) + (row_sums.reshape((-1, 1)) <= 1) * affine

        # To get the difference between the final two points and the starting point,
        # We need to slice the last 2 columns, and then subtract the reshaped array
        # Reshape ensures it does not become 1D, so maintains column subtraction.
        vector_differences = np.subtract(
            random_samples[:, 1:3], random_samples[:, 0].reshape(-1, 1))

        weighted_samples = random_samples[:, 0] + \
            np.sum(vector_differences * affine, axis=1)

        generated_points = [pt.coords for pt in weighted_samples]

        df = pd.DataFrame(generated_points, columns=self.labels)

        return df

    def construct_polygon(self, list_of_points: List[float]) -> EpiPolygon:
        my_polygon = EpiPolygon(*list_of_points)
        return my_polygon
