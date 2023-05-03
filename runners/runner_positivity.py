# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from epiverse.utilities.data_generation.data_generator_polygon import DataGeneratorPolygon
from epiverse.convex_hull.positivity_by_convex_hull import PositivityConvexHull

PLOT_HALFSPACE = False

if __name__ == "__main__":

    # Simulation steps:
    # 1. Generate 2D data for treatment groups.
    # 1a. Do for complete nonpositivity
    upper_triangle_data = DataGeneratorPolygon(
        ["C1", "C2"], [0, 0], [1.5, 1.5], [0, 1.5]).generate_data(n=100, seed=123)
    lower_triangle_data = DataGeneratorPolygon(
        ["C1", "C2"], [0, 0], [1.5, 0], [1.5, 1.5]).generate_data(n=100, seed=123)

    # 1b. Do for partial nonpositivity
    upper_trapezoid_data = DataGeneratorPolygon(
        ["C1", "C2"], [0, 2], [1, 0.5], [2, 0.5], [3, 2]).generate_data(n=100, seed=123)
    lower_trapezoid_data = DataGeneratorPolygon(
        ["C1", "C2"], [0, 0], [3, 0], [2, 1.5], [1, 1.5]).generate_data(n=100, seed=123)

    # 1c. Do for perfect positivity.
    dgp = DataGeneratorPolygon(["C1", "C2"], [0, 0], [2, 0], [2, 2], [0, 2])
    data1 = dgp.generate_data(n=100, seed=123)
    data2 = dgp.generate_data(n=100, seed=246)

    # 2. Compute Positivity Convex Hulls for the 2D data
    fig, ax = plt.subplots(nrows=3, ncols=3, sharey=True, sharex=True)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    pch_triangle = PositivityConvexHull(upper_triangle_data,
                                        lower_triangle_data,
                                        ["C1", "C2"])

    # Plot halfspaces of each data generating mechanism
    if PLOT_HALFSPACE:
        pch_triangle.plot_2d_halfspaces(
            axes=ax[0, 0], plot_treated=True, plot_untreated=False)
        pch_triangle.plot_2d_halfspaces(
            axes=ax[0, 2], plot_treated=False, plot_untreated=True)
    # Plot Upper Hull
    ax[0, 0].fill([0, 1.5, 0], [0, 1.5, 1.5], alpha=0.5)
    pch_triangle.plot_2d_convex_hull(
        axes=ax[0, 0], plot_treated=True, plot_untreated=False)
    ax[0, 0].scatter(upper_triangle_data["C1"],
                     upper_triangle_data["C2"], alpha=0.5, s=15)

    # Plot Lower Hull
    ax[0, 2].fill([0, 1.5, 1.5], [0, 0, 1.5], alpha=0.5)
    pch_triangle.plot_2d_convex_hull(
        axes=ax[0, 2], plot_treated=False, plot_untreated=True)
    ax[0, 2].scatter(lower_triangle_data["C1"],
                     lower_triangle_data["C2"], alpha=0.5, s=15)

    # Try for Trapezoidal data
    pch_trapezoid = PositivityConvexHull(
        upper_trapezoid_data, lower_trapezoid_data, ["C1", "C2"])

    if PLOT_HALFSPACE:
        pch_trapezoid.plot_2d_halfspaces(
            axes=ax[1, 0], plot_treated=True, plot_untreated=False)
        pch_trapezoid.plot_2d_halfspaces(
            axes=ax[1, 2], plot_treated=False, plot_untreated=True)

    pch_trapezoid.plot_2d_convex_hull(
        axes=ax[1, 1], alpha=0.1)
    pch_trapezoid.plot_2d_intersection(axes=ax[1, 1])

    pch_trapezoid.plot_2d_convex_hull(
        axes=ax[1, 0], plot_treated=True, plot_untreated=False)
    pch_trapezoid.plot_2d_convex_hull(
        axes=ax[1, 2], plot_treated=False, plot_untreated=True)
    ax[1, 0].scatter(upper_trapezoid_data["C1"],
                     upper_trapezoid_data["C2"], alpha=0.5, c="navy", s=15)
    ax[1, 2].scatter(lower_trapezoid_data["C1"],
                     lower_trapezoid_data["C2"], alpha=0.5, c="green", s=15)

    # Try for perfectly overlapped data
    pch_data = PositivityConvexHull(data1, data2, ["C1", "C2"])

    if PLOT_HALFSPACE:
        pch_data.plot_2d_halfspaces(
            axes=ax[2, 0], plot_treated=True, plot_untreated=False)
        pch_data.plot_2d_halfspaces(
            axes=ax[2, 2], plot_treated=False, plot_untreated=True)

    pch_data.plot_2d_convex_hull(ax[2, 1], alpha=0.1)
    pch_data.plot_2d_intersection(axes=ax[2, 1])

    pch_data.plot_2d_convex_hull(
        axes=ax[2, 0], plot_treated=True, plot_untreated=False)
    pch_data.plot_2d_convex_hull(
        axes=ax[2, 2], plot_treated=False, plot_untreated=True)
    ax[2, 0].scatter(data1["C1"],
                     data1["C2"], alpha=0.5, c="navy", s=15)
    ax[2, 2].scatter(data2["C1"],
                     data2["C2"], alpha=0.5, c="green", s=15)

    # Show Plot
    fig.supxlabel("C1")
    fig.supylabel("C2")
    ax[0, 0].set_title("Points and Convex Hull of Treated")
    ax[0, 1].set_title("Intersection of Convex Hulls")
    ax[0, 2].set_title("Points and Convex Hull of Untreated")
    plt.show()

    # 3. Compute distance for non-positive points
