"""Crosstalk Analysis Submodule.

Provides functions for analyzing and visualizing crosstalk between channels in imaging data.

Authors:
    - The SCALLOPS development team
"""

from collections.abc import Callable
from math import comb
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from xarray import DataArray

from scallops.reads import li_speed_slope
from scallops.visualize.utils import _figsize


def pairwise_channel_scatter_plot(
    bases: DataArray | pd.DataFrame,
    plot_func: Callable[[np.ndarray, np.ndarray, int, int, Axes, Any], None] = None,
    q: tuple[float, ...] = None,
    col_wrap: int = 3,
    intensity_column: str = "intensity",
    **plot_args,
) -> tuple[plt.Figure, plt.Axes]:
    """Pairwise channel intensity scatter plot.

    This function creates a scatter plot for pairwise combinations of channels in the provided DataArray.

    :param bases: The input array containing intensity values for different channels across samples and time points.
    :param plot_func:  Function to plot x-y scatter plot that accepts channel i values, channel j values, i, j, axis,
        and plot keyword arguments (e.g. markersize).
    :param q: Tuple of lower and upper quantiles to include for Li and Speed fit. If `None`, do not perform fit.
    :param col_wrap: Wrap the plot at this width, so that the plot spans multiple rows.
    :param intensity_column: The name of the column in `bases` containing the intensity when bases is data frame.
    :param plot_args: Additional arguments to pass to the plot function (e.g. markersize)
    :return: The figure and axes object with the plot drawn onto it.

    :examples:

        1. Generate a pairwise channel scatter plot

            .. code-block:: python

                import xarray as xr
                import matplotlib.pyplot as plt
                from scallops.visualize import pairwise_channel_scatter_plot

                # Create a sample DataArray
                channels = ["ChannelA", "ChannelT", "ChannelG", "ChannelC"]
                data = xr.DataArray(
                    np.random.rand(100, 10, len(channels)),
                    dims=("read", "t", "c"),
                    coords={"c": channels},
                )

                # Generate a pairwise channel scatter plot
                fig, axes = pairwise_channel_scatter_plot(
                    data, q=(0.1, 0.9), col_wrap=2
                )

                # Show the plot
                plt.show()

       2. Generate a pairwise channel scatter plot with colors by their base calls

            .. code-block:: python

                import xarray as xr
                import matplotlib.pyplot as plt
                import matplotlib.patches as mpatches
                from scallops.visualize import pairwise_channel_scatter_plot

                # Create a sample DataArray
                channels = ["ChannelA", "ChannelT", "ChannelG", "ChannelC"]
                data = xr.DataArray(
                    np.random.rand(100, 10, len(channels)),
                    dims=("read", "t", "c"),
                    coords={"c": channels},
                )

                bases = ["G", "T", "A", "C"]
                base_calls = pd.Series(np.random.choice(bases, size=100))
                base_colors = {"G": "green", "T": "red", "A": "magenta", "C": "cyan"}


                def plot_func(x, y, i, j, ax, **plot_args):
                    ax.scatter(
                        x, y, c=base_calls.apply(lambda x: base_colors[x]), **plot_args
                    )


                fig, axes = pairwise_channel_scatter_plot(
                    data.isel(t=0), plot_func=plot_func
                )
                patches = [
                    mpatches.Patch(color=color, label=label)
                    for label, color in base_colors.items()
                ]
                plt.legend(handles=patches)
                plt.show()
    """

    channels = (
        bases.c.values if isinstance(bases, xr.DataArray) else bases["c"].unique()
    )
    ncombs = comb(len(channels), 2)

    ncol = ncombs
    nrow = 1
    if col_wrap is not None and ncol > col_wrap:
        ncol = col_wrap
        nrow = int(np.ceil(ncombs / col_wrap))

    figsize = _figsize(nrow=nrow, ncol=ncol, aspect=1, size=5)
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=figsize, squeeze=False)

    _axes = axes
    axes = axes.ravel()
    max_slopes = []
    max_slope = 0
    k = 0

    def default_plot_func(x, y, i, j, ax, **plot_kwargs):
        ax.plot(x, y, ",", **plot_kwargs)

    default_plot_args = dict()
    if plot_func is None:
        default_plot_args.update(dict(markersize=0.6, alpha=0.7))
        plot_func = default_plot_func
    default_plot_args.update(**plot_args)
    for key in ["markersize", "ms", "s"]:
        marker_size = default_plot_args.get(key)
        if marker_size is not None:
            break
    if marker_size is None:
        marker_size = 0.6
    for i in range(len(channels)):
        x = (
            bases.isel(c=i).data.flatten()
            if isinstance(bases, xr.DataArray)
            else bases[bases["c"] == channels[i]][intensity_column]
        )
        for j in range(i):
            y = (
                bases.isel(c=j).data.flatten()
                if isinstance(bases, xr.DataArray)
                else bases[bases["c"] == channels[j]][intensity_column]
            )
            ax = axes[k]
            plot_func(x, y, i, j, ax, **default_plot_args)

            if q is not None:
                df, x_slope = li_speed_slope(x, y, q)
                max_slope = max(max_slope, abs(x_slope))
                axis_range = df.x.min(), df.x.max()
                axis_range = (
                    min(axis_range[0], df.y_pred.min()),
                    max(axis_range[1], df.y_pred.max()),
                )

                # line along y-axis
                ax.plot(df.x, df.y_pred, color="black", linewidth=1)
                # points used for fit
                ax.scatter(df.x, df.y, c="black", s=marker_size, alpha=0.7)

                df, y_slope = li_speed_slope(y, x, q)
                max_slope = max(max_slope, abs(y_slope))

                # line along x-axis
                ax.plot(df.y_pred, df.x, color="black", linewidth=1)
                # points used for fit
                ax.scatter(df.y, df.x, c="black", s=marker_size, alpha=0.7)

                axis_range = (
                    min(axis_range[0], df.y_pred.min()),
                    max(axis_range[1], df.y_pred.max()),
                )
                axis_range = (
                    min(axis_range[0], df.x.min()),
                    max(axis_range[1], df.x.max()),
                )

                pad = (axis_range[1] - axis_range[0]) * 0.05
                axis_range = list(axis_range)
                axis_range[0] = axis_range[0] - pad
                axis_range[1] = axis_range[1] + pad
                ax.set_xlabel(f"{channels[i]}, slope: {x_slope:.3f}", rotation=0)
                ax.set_ylabel(f"{channels[j]}, slope: {y_slope:.3f}", rotation=90)
                ax.set_xlim(axis_range)
                ax.set_ylim(axis_range)

            else:
                ax.set_xlabel(f"{channels[i]}", rotation=0)
                ax.set_ylabel(f"{channels[j]}", rotation=90)

            k += 1
    max_slopes.append(max_slope)
    if q is not None:
        title = []
        for i in range(len(max_slopes)):
            title.append(f"cc #:{max_slopes[i]:.3f}")
        fig.suptitle(", ".join(title))

    fig.tight_layout()
    return fig, _axes
