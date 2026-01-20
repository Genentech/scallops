"""Histogram Generation Submodule.

Provides functions for generating histograms from imaging data.



Authors:
    - The SCALLOPS development team
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from xarray import DataArray

from scallops.experiment.elements import Experiment
from scallops.experiment.util import _concat_images
from scallops.visualize.utils import _figsize


def channel_hist_plot(
    image: Experiment | DataArray,
    binrange: tuple[float, float] = None,
    bins: str | int | Sequence[float] = "auto",
    height: float = 5,
    aspect: float = 1,
    **kwargs: dict,
) -> Axes:
    """Plot histogram per image/channel.

    :param height:
        Height (in inches) of each facet.
    :param aspect:
        Aspect ratio of each facet, so that aspect * height gives the width of each facet in inches
    :param image:
        XArray with the dimensions (t,c,z,y,c) or (i,t,c,z,y,c)
    :param binrange:
        Lowest and highest value for bin edges. Defaults to data extremes.
    :param bins: Generic bin parameter that can be the name of a reference rule, the number of bins, or the breaks of
                 the bins. Passed to :meth:`numpy.histogram_bin_edges`
    :param kwargs:
        Additional keyword arguments to pass to :meth:`matplotlib.pyplot.hist`
    :return:
        The matplotlib axes containing the plot.

    :example:

        .. code-block:: python

            import numpy as np
            import xarray as xr
            from scallops.visualize import channel_hist_plot

            # Generate synthetic data
            data = np.random.normal(size=(10, 3, 64, 64))
            image = xr.DataArray(data, dims=("t", "c", "y", "x"))

            # Plot histogram
            channel_hist_plot(image, binrange=(-3, 3), bins=50, height=8, aspect=2)
            plt.show()
    """
    if isinstance(image, Experiment):
        image = _concat_images(image)
    if "t" in image.sizes:
        image = image.isel(t=0)
    if binrange is not None:
        assert len(binrange) == 2, "Length of bin_range must be 2"
    channels = image.c.values
    nchannels = len(channels)
    nimages = image.sizes["i"] if "i" in image.dims else 1
    images = image.i.values if "i" in image.dims else [""]
    figsize = _figsize(ncol=nimages, nrow=1, aspect=aspect, size=height)

    fig, axes = plt.subplots(1, nimages, figsize=figsize, squeeze=False, sharey=True)
    if binrange is None:
        binrange = image.min().values, image.max().values
    bins = np.histogram_bin_edges(image, bins=bins, range=binrange)
    hist_kwargs = dict(alpha=0.5)
    hist_kwargs.update(kwargs)
    for i in range(nimages):
        ax = axes[0, i]
        ax.set_title(images[i])
        for j in range(nchannels):
            vals = image.isel(c=j, i=i, missing_dims="ignore").values.flatten()
            c = channels[j]
            _ = ax.hist(vals, bins=bins, label=str(c), **hist_kwargs)

    ax.legend(bbox_to_anchor=(1.1, 1.00))
    fig.tight_layout()
    return ax


def in_situ_barcode_hist_plot(
    reads_df: pd.DataFrame,
    counts: Sequence[int] | None = (0, 1, 2, 3, 4, 5),
    hue: str = None,
    normalize: bool = True,
    **fig_kw,
) -> plt.Axes:
    """Generate a histogram plot depicting the percentage of cells containing barcode reads. This
    function takes a DataFrame with read tables and plots a histogram showing the distribution of
    cells based on the number of barcode reads.

    :param reads_df: DataFrame containing read tables from the output reads folder.
    :param counts: Optional list of histogram bins.
    :param hue: Grouping variable for the second layer of grouping.
    :param normalize: Whether to normalize counts.
    :param fig_kw: Additional keyword arguments passed to `.pyplot.figure` call.
    :return: Returns the Axes object with the plot drawn onto it.

    :example:

        .. code-block:: python

            from scallops.visualize import in_situ_barcode_hist_plot
            import pandas as pd
            import numpy as np

            data = {
                "well": np.random.choice(["A", "B", "C", "D"], size=100),
                "label": np.random.choice(range(1, 10), size=100),
                "barcode_match": np.random.choice([True, False], size=100),
            }
            reads_df = pd.DataFrame(data)
            in_situ_barcode_hist_plot(
                reads_df, counts=[0, 1, 2, 3], hue="well", normalize=True
            )
    """

    def _create_df(df):
        cols = [c for c in ["well", "tile"] if c in df.columns]
        barcode_exact_match_sum_df = (
            df.query("label>0")[cols + ["label", "barcode_match"]]
            .groupby(cols + ["label"])
            .sum()
        )
        barcode_exact_match_counts = barcode_exact_match_sum_df[
            "barcode_match"
        ].value_counts()
        if counts is not None:
            cells = np.zeros(len(counts))
            for i in range(len(counts) - 1):
                cells[i] = (
                    barcode_exact_match_counts[i]
                    if i in barcode_exact_match_counts
                    else 0
                )
            cells[len(cells) - 1] = barcode_exact_match_counts[
                barcode_exact_match_counts.index >= counts[len(counts) - 1]
            ].sum()
            counts_str = np.array(counts).astype(str)
            counts_str[len(counts_str) - 1] += "+"
            df_hist = pd.DataFrame({"cat": counts_str, "cells": cells})
        else:
            df_hist = pd.DataFrame(
                {
                    "cat": barcode_exact_match_counts.index,
                    "cells": barcode_exact_match_counts.values,
                }
            )
        if normalize:
            df_hist["value"] = (df_hist["cells"] / df_hist["cells"].sum()) * 100.0
        else:
            df_hist["value"] = df_hist["cells"]
        return df_hist

    if hue is None:
        df_hist = _create_df(reads_df)
    else:
        df_hists = []
        for key, grouped_df in reads_df.groupby(hue):
            df_hist = _create_df(grouped_df)
            df_hist[hue] = key
            df_hists.append(df_hist)
        df_hist = pd.concat(df_hists)
    fig = plt.figure(**fig_kw)
    ax = fig.add_subplot(1, 1, 1)
    sns.barplot(x="cat", y="value", hue=hue, data=df_hist, ax=ax)
    ax.set_xlabel("Reads per label\n(exact match)")
    ax.set_ylabel("Labels" + ("(%)" if normalize else ""))
    return ax
