"""Scallop tools for visualizing registration.

Provides functions for visualizing registration results, allowing users to diagnose and plot images from
different rounds or channels for quality assessment.



Authors:
    - The SCALLOPS development team
"""

from collections.abc import Sequence
from itertools import cycle, product
from typing import Any

from matplotlib import pyplot as plt
from matplotlib import rcParams
from numpy import array_equal
from xarray import DataArray

from scallops.visualize.imshow import imshow_plane
from scallops.visualize.utils import _linear_alpha_cmap


def diagnose_registration(
    imagestack: DataArray,
    *sels: None | Sequence[dict[str, Any]],
    ax: None | plt.Axes = None,
    title: None | str = None | str,
    **kwargs,
) -> Sequence[plt.Axes]:
    """Overlays 2-d images extracted from a DataArray for visualizing alignment of images from
    different rounds or channels selected with the `sel` parameter. Up to six images can be selected
    and shown in different colors. The same `Axes.X` and `Axes.Y` indices should be used for every
    Selector.

    :param imagestack:
        DataArray from which to extract 2-d images for plotting.
    :param sels:
        Optional list, but only if `imagestack` is already of shape (1, 1, 1, y, x). Selectors to pass
        `imagestack.sel`, selects the (y, x) planes to be plotted.
    :param ax:
        Axes to plot on. If not passed, defaults to the current axes.
    :param title:
        Title to assign the Axes being plotted on.
    :param kwargs:
        Additional keyword arguments to pass to `imshow_plane`.

    :return:
        List of matplotlib Axes representing the overlayed images.

    :example:

        .. code-block:: python

            import numpy as np
            import xarray as xr
            from scallops.visualize.registration import diagnose_registration

            # Create synthetic DataArray
            image_shape = (3, 4, 1, 100, 100)  # (t, c, z, y, x)
            imagestack = xr.DataArray(
                np.random.rand(*image_shape), dims=("t", "c", "z", "y", "x")
            )

            # Define selectors
            selectors = [
                {"t": 0, "c": 0},
                {"t": 1, "c": 1},
                {"t": 2, "c": 2},
            ]

            # Plot the registration diagnosis
            axes = diagnose_registration(
                imagestack.squeeze(), *selectors, title="Registration Diagnosis"
            )
            plt.show()
    """
    if ax is None:
        ax = plt.gca()

    if title is not None:
        ax.set_title(title)

    # add linear alpha to existing colormaps to avoid the "white" color of the
    # map at 0 from clobbering previously plotted images. Functionally this
    # enables high intensity spots to be co-visualized in the same frame.
    cmaps = [
        plt.cm.Blues,
        plt.cm.Reds,
        plt.cm.Greens,
        plt.cm.Purples,
        plt.cm.Greys,
        plt.cm.Oranges,
    ]

    alpha_cmap_cycle = cycle([_linear_alpha_cmap(cm) for cm in cmaps])

    all_axes = [
        imshow_plane(imagestack.sel(**sel), ax=ax, cmap=cmap, **kwargs)
        for sel, cmap in zip(sels, alpha_cmap_cycle)
    ]
    return all_axes


def plot_registration(
    stacks: Sequence[DataArray],
    dpi: int = 250,
    titles: None | Sequence[str] = None,
    plot_cols: None | int = None,
    **kwargs,
) -> Sequence[plt.Axes]:
    """Plot raw images from a list of DataArray with registered images to be visualized.

    :param stacks:
        List of DataArray with registered images to be visualized.
    :param dpi:
        Requested DPI (only relevant if saving to file or visualizing in Jupyter notebooks).
    :param titles:
        Optional list of titles. Must be the same length as the number of stacks to plot.
    :param plot_cols:
        Number of columns to plot.
    :param kwargs:
        Key word arguments to be passed to plt.subplots

    :return:
        List of matplotlib Axes representing the plotted images.

    :example:

        .. code-block:: python

            import numpy as np
            import xarray as xr
            from scallops.visualize.registration import plot_registration

            # Create synthetic DataArrays
            image_shape = (3, 4, 100, 100)  # (t, c, y, x)
            stacks = [
                xr.DataArray(np.random.rand(*image_shape), dims=("t", "c", "y", "x")),
                xr.DataArray(np.random.rand(*image_shape), dims=("t", "c", "y", "x")),
                xr.DataArray(np.random.rand(*image_shape), dims=("t", "c", "y", "x")),
            ]

            # Plot the registration results
            axes = plot_registration(
                stacks, dpi=150, titles=["Stack 1", "Stack 2", "Stack 3"]
            )
            plt.show()
    """
    stacks = [x.isel(z=0, missing_dims="ignore").squeeze() for x in stacks]
    axes = None
    all_rounds = [stack.coords["t"].values for stack in stacks]
    assert all(array_equal(all_rounds[0], x) for x in all_rounds), (
        "Images differ in the number of cycles"
    )
    rounds = all_rounds[0]
    all_channels = [stack.coords["c"].values for stack in stacks]
    assert all(array_equal(all_channels[0], x) for x in all_channels), (
        "Images differ in the number of channels"
    )
    channels = all_channels[0]
    assert isinstance(stacks, list), "stacks must be a list of stacks of len >= 1"
    if titles is not None:
        assert len(stacks) == len(titles), "Titles should have the same shape as stacks"
    rcParams["figure.dpi"] = dpi
    nstacks = len(stacks)
    prod = product(rounds, channels)
    args = [{"t": cy, "c": channel} for cy, channel in prod]
    if nstacks == 1:
        axes = diagnose_registration(stacks[0], *args)
    else:
        if plot_cols is not None:
            ncols = plot_cols
        else:
            ncols = min(3, nstacks)
        d = nstacks // ncols
        nrows = 1 if d <= 1 else d
        f, axarr = plt.subplots(nrows=nrows, ncols=ncols, **kwargs)
        for i, (stack, ax) in enumerate(zip(stacks, axarr.flat)):
            title = titles[i] if titles is not None else None
            axes = diagnose_registration(stack, *args, title=title, ax=ax)
    return axes
