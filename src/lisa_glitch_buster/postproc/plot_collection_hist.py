import matplotlib.pyplot as plt
import numpy as np


def hist_collection(x, bins, ax):
    """
    Plot a histogram with collection bins for values outside the specified range.

    Parameters:
    - x: array-like, input data
    - bin_range: tuple, (min, max) for the main histogram range
    - n_bins: int, number of bins for the main histogram
    - ax: matplotlib.axes.Axes object to plot on
    """

    # Count values in the collection bins
    below_range = np.sum(x < bins[0])
    above_range = np.sum(x > bins[-1])

    # Create histogram data for the main range
    hist, bin_edges = np.histogram(x, bins=bins)
    bin_range = (bins[0], bins[-1])

    # Add collection bins to the histogram data
    hist = np.concatenate(([below_range], hist, [above_range]))
    bin_edges = np.concatenate(
        ([bin_range[0] - 1], bin_edges, [bin_range[1] + 1])
    )

    # Plot the histogram
    ax.bar(
        bin_edges[:-1],
        hist,
        width=np.diff(bin_edges),
        align="edge",
        edgecolor="black",
    )

    # # Adjust x-axis to show collection bins clearly
    # ax.set_xlim(bin_range[0] - 5, bin_range[1] + 5)

    # Add labels for collection bins
    ax.text(
        bin_range[0] - 2.5,
        below_range / 2,
        f"<{bin_range[0]}\n({below_range})",
        ha="center",
        va="center",
    )
    ax.text(
        bin_range[1] + 2.5,
        above_range / 2,
        f">{bin_range[1]}\n({above_range})",
        ha="center",
        va="center",
    )
    return hist, bin_edges
