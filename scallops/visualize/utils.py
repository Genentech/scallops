"""Utility functions for Visualization.

Provides utility functions for visualization.

Authors:
    - The SCALLOPS development team
"""

from itertools import groupby
from typing import Literal, Sequence

import dask.array as da
import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap, ListedColormap
from pydantic import ValidationError
from xarray import DataArray


def _get_image_crop_slice(
    indices: np.ndarray, image_size: int, crop_size: int, discard_small_crops: bool
) -> slice | None:
    """Create slice for label indices.

    :param indices: Array of label indices
    :param image_size: Size of the image
    :param crop_size: Size of crop
    :param discard_small_crops: Discard small crops at the image edges
    :return: Slice for image crop or None if `discard_small_crops` is `True` and crop
        is too small
    """
    idx_mean = indices.mean()
    idx_min = max(0, int(np.round(idx_mean - crop_size / 2)))
    idx_max = min(image_size, int(np.round(idx_mean + crop_size / 2)))
    remainder = crop_size - (idx_max - idx_min)  # grow or shrink
    if remainder != 0:
        if discard_small_crops:
            return None
        idx_min = max(0, idx_min - remainder)
        remainder = crop_size - (idx_max - idx_min)
        if remainder != 0:  # try idx_max if unable to expand idx_min
            idx_max = min(image_size, idx_max + remainder)
    assert idx_max - idx_min == crop_size, f"{idx_min}-{idx_max} != {crop_size}"
    return slice(idx_min, idx_max)


def _figsize(
    nrow: int = 1, ncol: int = 1, aspect: float = 1, size: float = 3
) -> tuple[float, float]:
    """Calculate figure size for plots using rows, columns, and aspect ratio.

    :param nrow: Number of rows in the figure grid (default is 1).
    :param ncol: Number of columns in the figure grid (default is 1).
    :param aspect: Aspect ratio of the plot (default is 1).
    :param size: Base size for scaling (default is 3).
    :return: A tuple representing the width and height of the figure.
    """
    return ncol * size * aspect, nrow * size


def _wrap_cols(ncol: int, col_wrap: int) -> tuple[int, int]:
    """Adjust the number of columns and rows to wrap columns based on the specified
    col_wrap.

    :param ncol: Original number of columns.
    :param col_wrap: Maximum number of columns before wrapping to a new row.
    :return: A tuple with the adjusted number of columns and rows.
    """
    nrow = 1
    if ncol > col_wrap:
        _ncol = ncol
        ncol = col_wrap
        nrow = int(np.ceil(_ncol / col_wrap))
    return ncol, nrow


def _create_color_map_for_rgb(rgb: np.ndarray | list[float]) -> ListedColormap:
    """Create a colormap from an RGB array of length 3.

    :param rgb: An RGB array or list with three float values (one per channel).
    :return: A ListedColormap based on the input RGB values.
    """
    vals = np.linspace(0, 1, 256)
    cols = [np.interp(vals, [0, 1], [0, rgb[i]]) for i in range(3)]
    return ListedColormap(np.stack(cols, axis=1))


def _linear_alpha_cmap(cmap: Colormap) -> ListedColormap:
    """Add a linear alpha channel to an existing colormap.

    :param cmap: Input colormap.
    :return: A new ListedColormap with a linear alpha channel.
    """
    alpha_cmap = cmap(np.linspace(0, 1, cmap.N))
    alpha_cmap[:, -1] = np.linspace(0, 1, cmap.N)  # Set linear alpha channel
    return ListedColormap(alpha_cmap)


def channel_thresholds(
    image: DataArray,
    percentile_min: Sequence[float] | float = 0.0,
    percentile_max: Sequence[float] | float = 99.0,
    pad_min: Sequence[float] | float = 0,
    pad_max: Sequence[float] | float = 0,
    thresholds: dict[int, tuple[float, float]] = None,
) -> dict[int, tuple[float, float]]:
    """Compute thresholds per channel for contrast adjustment in visualization.

    :param image: XArray with dimensions (t,c,z,y,x) or (i,t,c,z,y,x)
    :param percentile_min: Determines the minimal threshold for contrast adjustment.
        If a list, apply different percentiles per channel. If a float, apply the same
        percentile for all channels.
    :param percentile_max: Determines the maximal threshold for contrast adjustment.
        If a list, apply different percentiles per channel. If a float, apply the same
        percentile for all channels.
    :param pad_min: Adjust the minimum value for visualization. If a list, apply
        different pads per channel. If a float, apply the same pad for all channels.
    :param pad_max: Adjust the maximum value for visualization. If a list, apply
        different pads per channel. If a float, apply the same pad for all channels.
    :param thresholds: Optional precomputed dictionary mapping channel to tuple of
        (tmin, tmax).

    :return: A dictionary that maps channel to tuple of (tmin, tmax) for contrast
        adjustment.

    :example:

        .. code-block:: python

            import numpy as np
            import xarray as xr
            from scallops.visualize.utils import channel_thresholds

            # Create synthetic data
            channels, z, y, x = 3, 10, 100, 100
            synthetic_data = np.random.rand(channels, z, y, x)
            synthetic_image = xr.DataArray(synthetic_data, dims=("c", "z", "y", "x"))

            # Compute thresholds
            thresholds = channel_thresholds(
                synthetic_image, percentile_min=1, percentile_max=99
            )

            print(thresholds)
    """

    channels = image.c.values
    nchannels = len(channels)
    is_percentile_min_sequence = False
    if isinstance(percentile_min, Sequence):
        is_percentile_min_sequence = True
        assert len(percentile_min) == nchannels
    is_percentile_max_sequence = False
    if isinstance(percentile_max, Sequence):
        is_percentile_max_sequence = True
        assert len(percentile_max) == nchannels
    _thresholds = dict() if thresholds is None else thresholds.copy()

    for channel in channels:
        if channel not in _thresholds:
            q_min = (
                percentile_min[channel]
                if is_percentile_min_sequence
                else percentile_min
            )
            q_max = (
                percentile_max[channel]
                if is_percentile_max_sequence
                else percentile_max
            )
            data = image.sel(c=channel).data
            q = (
                da.percentile(data.flatten(), [q_min, q_max])
                if isinstance(data, da.Array)
                else np.percentile(data, [q_min, q_max])
            )
            tmin = q[0] - pad_min
            tmax = q[1] + pad_max
            _thresholds[channel] = (tmin, tmax)
    return _thresholds


def _coords(exp, key: str) -> tuple[None | str, np.ndarray]:
    """Extract well and tile coordinates from an experiment image metadata.

    Attempts to parse a key to extract the well and tile, and then retrieves the
    position coordinates from the OME metadata.

    :param exp: An experiment object containing image metadata.
    :param key: The image key in the format 'well-tile'.
    :return: A tuple containing the well identifier and an array of [x, y] coordinates.
    """
    c = 0
    while c < 3:
        c += 1
        try:
            well, tile = key.split("-")
            im = exp.images[key]
            if im is None:
                return None, np.array([np.nan, np.nan])
            pixels = im.attrs["ome"]["images"][0]["pixels"]
            plane = pixels["planes"][0]
            return well, np.array(
                [plane["position_x"], plane["position_y"]], dtype=float
            )
        except ValidationError:
            return None, np.array([np.nan, np.nan])
    return None, np.array([np.nan, np.nan])


def _infer_positions(exp, max_workers: int = -1) -> dict[str, list[int]]:
    """Infer the well positions from image metadata in an experiment object.

    This function uses parallel processing to extract coordinates for each tile and
    groups them by well to infer the structure of the tiles.

    :param exp: An experiment object containing image metadata.
    :param max_workers: Number of parallel workers (default is -1, which uses all
        available cores).
    :return: A dictionary where keys are well identifiers and values are lists of row
        counts.
    """
    wells, *coords = zip(
        *Parallel(n_jobs=max_workers)(
            delayed(_coords)(exp, tile) for tile in exp.images.keys()
        )
    )

    def _get_structure(coords_: list[tuple[str, np.ndarray]]) -> tuple[str, list[int]]:
        """Compute the tile structure for a given well.

        :param coords_: List of tuples containing well identifiers and their
            coordinates.
        :return: A tuple of the well identifier and a list of row counts for each tile
            range.
        """
        well, arr = zip(*coords_)
        sw = list(set(well))
        assert len(sw) == 1, "Well grouping failed"
        try:
            arr = np.array(arr, dtype=float)
            nones = np.isnan(arr).any(axis=1)
            arr = arr[~nones]
            arr -= arr.min(axis=0)
        except ValueError as err:
            raise err

        arr //= np.median(np.abs(np.diff(arr, axis=0)), axis=0)
        changespp = np.where(np.abs(np.diff(arr[:, 1])) > 10)[0] + 1
        ranges = (
            [(0, changespp[0])]
            + list(zip(changespp, changespp[1:]))
            + [(changespp[-1],)]
        )

        return sw[0], [
            arr[ixs[0] : ixs[1], :].shape[0]
            if len(ixs) == 2
            else arr[ixs[0] :, :].shape[0]
            for ixs in ranges
        ]

    groupper = groupby(
        [y for y in zip(wells, coords[0]) if y[0] is not None], lambda x: x[0]
    )
    return {
        well: _get_structure(list(gr))[1] for well, gr in groupper if well is not None
    }


def multicolor_labels(
    ax: plt.Axes,
    list_of_labels: Sequence[str],
    list_of_colors: Sequence[str],
    axis: Literal["x", "y", "both"] = "x",
    anchorpad: float = 0,
    **kwargs: dict,
) -> None:
    """Create axes labels with multiple colors.

    :param ax: matplotlib.axes.Axes
        The axes object where the labels should be drawn.
    :param list_of_labels: list
        A list of all text items.
    :param list_of_colors: list
        A corresponding list of colors for the strings.
    :param axis: {'x', 'y', 'both'}, optional
        Specifies which label(s) should be drawn.
    :param anchorpad: float, optional
        Padding between the label and the axis.
    :param kwargs: dict, optional
        Additional keyword arguments passed to textprops.

    :example:

        .. code-block:: python

            import matplotlib.pyplot as plt
            from scallops.visualize.utils import multicolor_labels

            fig, ax = plt.subplots()
            strings = ["Label 1", "Label 2", "Label 3"]
            colors = ["red", "green", "blue"]
            multicolor_labels(ax, strings, colors, axis="y")
            plt.show()
    """
    from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker

    # x-axis label
    if axis == "x" or axis == "both":
        boxes = [
            TextArea(
                text, textprops=dict(color=color, ha="left", va="bottom", **kwargs)
            )
            for text, color in zip(list_of_labels, list_of_colors)
        ]
        xbox = HPacker(children=boxes, align="center", pad=0, sep=5)
        anchored_xbox = AnchoredOffsetbox(
            loc=3,
            child=xbox,
            pad=anchorpad,
            frameon=False,
            bbox_to_anchor=(0.15, -0.09),
            bbox_transform=ax.transAxes,
            borderpad=0.0,
        )
        ax.add_artist(anchored_xbox)

    # y-axis label
    if axis == "y" or axis == "both":
        boxes = [
            TextArea(
                text,
                textprops=dict(
                    color=color, ha="left", va="bottom", rotation=90, **kwargs
                ),
            )
            for text, color in zip(list_of_labels, list_of_colors)
        ]
        ybox = VPacker(children=boxes, align="center", pad=0, sep=5)
        anchored_ybox = AnchoredOffsetbox(
            loc=3,
            child=ybox,
            pad=anchorpad,
            frameon=False,
            bbox_to_anchor=(-0.10, 0.2),
            bbox_transform=ax.transAxes,
            borderpad=0.0,
        )
        ax.add_artist(anchored_ybox)
