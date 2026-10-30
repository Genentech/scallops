"""Crosstalk Analysis Submodule.

Provides functions for analyzing and visualizing crosstalk between channels in imaging data.

Authors:
    - The SCALLOPS development team
"""

from collections.abc import Callable
from math import comb
from typing import Literal

import dask
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from xarray import DataArray

from scallops.visualize.utils import _figsize


def _pairwise_histograms(
    values: list, pairs: list[tuple[int, int]], bins: int
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute 2-d histograms for pairs of channels.

    Dask arrays take two passes: one for the per-channel ranges and one for all
    histograms, so each chunk is read once per pass.

    :param values: Flattened intensities for each channel.
    :param pairs: Pairs of channel indices (i, j); i is binned along x.
    :param bins: Number of bins per axis.
    :return: Mapping from (i, j) to counts, x edges, and y edges. Pairs where either
        channel has no finite values are omitted.
    """
    is_dask = any(isinstance(v, da.Array) for v in values)
    if is_dask:
        values = [da.asarray(v) for v in values]
        values = [v.rechunk(values[0].chunks) for v in values]
    xp = da if is_dask else np
    limits = [
        (xp.nanmin(v), xp.nanmax(v)) if v.size > 0 else (np.nan, np.nan) for v in values
    ]
    if is_dask:
        limits = dask.compute(limits)[0]
    ranges: list[tuple[float, float] | None] = []
    for lo, hi in limits:
        lo, hi = float(lo), float(hi)
        if np.isfinite(lo) and np.isfinite(hi):
            ranges.append((lo, hi if hi > lo else lo + 1))
        else:
            ranges.append(None)
    histograms = {}
    for i, j in pairs:
        range_i, range_j = ranges[i], ranges[j]
        if range_i is not None and range_j is not None:
            # values outside the range, including NaN, are not counted
            histograms[(i, j)] = xp.histogram2d(
                values[i], values[j], bins=bins, range=(range_i, range_j)
            )
    if is_dask:
        histograms = dask.compute(histograms)[0]
    return histograms


def pairwise_channel_scatter_plot(
    bases: DataArray | pd.DataFrame,
    plot_func: Callable[..., None] | None = None,
    col_wrap: int = 3,
    intensity_column: str = "intensity",
    kind: Literal["scatter", "hist"] = "hist",
    **plot_args,
) -> tuple[plt.Figure, plt.Axes]:
    """Pairwise channel intensity scatter plot.

    This function creates a scatter plot for pairwise combinations of channels in the provided DataArray.

    :param bases: The input array containing intensity values for different channels across samples and time points.
    :param plot_func:  Function to plot x-y scatter plot that accepts channel i values, channel j values, i, j, axis,
        and plot keyword arguments (e.g. markersize).
    :param col_wrap: Wrap the plot at this width, so that the plot spans multiple rows.
    :param intensity_column: The name of the column in `bases` containing the intensity when bases is data frame.
    :param kind: Plot every point (`scatter`) or a 2-d histogram with log-scaled counts
        (`hist`). Additional `plot_args` for `hist` are `bins` (default 256), `cmap`,
        `norm` (default :class:`~matplotlib.colors.LogNorm`), and arguments to
        :meth:`matplotlib.axes.Axes.pcolormesh`. Each channel uses the same
        range in every panel. For dask-backed input, `hist` makes two passes over the
        data (channel ranges, then all histograms) without loading it into memory.
        Ignored if `plot_func` is provided.
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
                fig, axes = pairwise_channel_scatter_plot(data, col_wrap=2)

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

    if kind not in ("scatter", "hist"):
        raise ValueError(f"`kind` must be 'scatter' or 'hist', got {kind!r}")
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

    values = [
        (
            bases.isel(c=i).data.flatten()
            if isinstance(bases, xr.DataArray)
            else bases[bases["c"] == channels[i]][intensity_column]
        )
        for i in range(len(channels))
    ]
    pairs = [(i, j) for i in range(len(channels)) for j in range(i)]

    histograms = None
    hist_norm = None
    if plot_func is None:
        if kind == "hist":
            plot_args = dict(plot_args)
            bins = plot_args.pop("bins", 256)
            plot_args.setdefault("cmap", "viridis")
            hist_norm = plot_args.pop("norm", None)
            histograms = _pairwise_histograms(values, pairs, bins)
        else:
            plot_args = dict(markersize=0.6, alpha=0.7) | plot_args

            def plot_func(x, y, i, j, ax, **plot_kwargs):
                ax.plot(x, y, ",", **plot_kwargs)

    for ax, (i, j) in zip(axes.ravel(), pairs):
        if histograms is not None:
            if (i, j) in histograms:
                counts, x_edges, y_edges = histograms[(i, j)]
                if counts.max() > 0:
                    ax.pcolormesh(
                        x_edges,
                        y_edges,
                        counts.T,
                        # log scale so sparse single-channel arms show next to the
                        # dense core; new instance per panel so each autoscales
                        norm=hist_norm if hist_norm is not None else LogNorm(),
                        rasterized=True,
                        **plot_args,
                    )
        else:
            assert plot_func is not None
            plot_func(values[i], values[j], i, j, ax, **plot_args)
        ax.set_xlabel(f"{channels[i]}", rotation=0)
        ax.set_ylabel(f"{channels[j]}", rotation=90)

    fig.tight_layout()
    return fig, axes
