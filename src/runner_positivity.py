# Imports
import numpy as np
import pandas as pd
from convex_hull.positivity_by_convex_hull import PositivityConvexHull

if __name__ == "__main__":

    # Simulation steps:
    # 1. Generate 2D data for treatment groups.
    # 1a. Do for complete nonpositivity
    # 1b. Do for partial nonpositivity
    # 1c. Do for positivity.
    # 2. Compute Positivity Convex Hulls for the 2D data
    # 3. Compute distance for non-positive points
    print("Hello World")
    print(np.linspace(0, 10, 200))
