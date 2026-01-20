"""Scallop's module with utilities to plot segmentation labels.

Provides functions for visualizing basic segmentation results, allowing users to plot
segmented images with optional titles and customization options.

Authors:
    - The SCALLOPS development team
"""

from typing import Sequence

from matplotlib import colors
from matplotlib import pyplot as plt
from numpy import ndarray
from skimage.util import img_as_uint


def plot_segmentation(
    data: Sequence[ndarray],
    dpi: int = 300,
    titles: None | Sequence[str] = None,
    save_to_file: None | str = None,
    plot_cols: None | int = None,
    fontsize: None | int = None,
    **kwargs,
):
    """Plot basic segmentation results.

    :param data: List of segmented images as ndarrays.
    :param dpi: Requested DPI (only relevant if saving to file).
    :param titles: Optional list of titles. Must be the same length as the number of
        stacks to  plot.
    :param save_to_file: Optional filename of the output file (format will be taken
        from the extension).
    :param plot_cols: Number of columns to plot.
    :param fontsize: Size of the titles' font.
    :param kwargs: Extra keyword arguments for plt.subplots.Axes.


    :example:

    .. code-block:: python

        import numpy as np
        from scallops.visualize.segmentation import plot_segmentation

        # Create synthetic segmented images
        segmentations = [
            np.random.rand(100, 100) > 0.5,
            np.random.rand(100, 100) > 0.5,
            np.random.rand(100, 100) > 0.5,
        ]

        # Plot segmentation results
        plot_segmentation(
            data=segmentations,
            dpi=150,
            titles=["Segmentation 1", "Segmentation 2", "Segmentation 3"],
            fontsize=12,
        )
        plt.show()
    """
    assert isinstance(data, list), "stacks must be a list of stacks of len >= 1"
    if titles is not None:
        assert len(data) == len(titles), "Titles should have the same shape as stacks"
    if dpi is not None:
        plt.rcParams["figure.dpi"] = dpi
    if fontsize is not None:
        plt.rcParams["font.size"] = fontsize
    cmap = colors.ListedColormap(["DarkBlue", "yellow"])
    norm = colors.BoundaryNorm([0, 10], cmap.N)
    nimgs = len(data)
    if nimgs == 1:
        plt.imshow(data[0])
    else:
        if plot_cols is not None:
            ncols = plot_cols
        else:
            ncols = min(3, nimgs)
        d = nimgs // ncols
        nrows = 1 if d <= 1 else d
        f, axarr = plt.subplots(nrows, ncols, constrained_layout=True)
        zipped = zip(data, axarr.flatten())
        for i, (stack, ax) in enumerate(zipped):
            title = titles[i] if titles is not None else None
            ax.set_title(title, fontsize=fontsize)
            ax.set_axis_off()
            ax.imshow(img_as_uint(stack), cmap=cmap, norm=norm, **kwargs)
    if save_to_file is not None:
        plt.savefig(save_to_file)
