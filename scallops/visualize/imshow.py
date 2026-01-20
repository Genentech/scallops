"""Image Display Submodule.

Provides functions for displaying images and overlays.


Authors:
    - The SCALLOPS development team
"""

import logging
import warnings
from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy
from itertools import chain, cycle, groupby
from random import choice
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import patches
from matplotlib import pyplot as plt
from natsort import natsorted
from numpy import ndarray
from pydantic import ValidationError
from skimage.measure import regionprops_table
from skimage.transform import resize
from xarray import DataArray

from scallops.experiment.elements import Experiment
from scallops.io import read_image
from scallops.visualize.grid_layout import _grid_indices, _well_grid
from scallops.visualize.utils import _infer_positions

logger = logging.getLogger("scallops")


def imshow_plane(
    image: ndarray | DataArray,
    ax: None | plt.Axes = None,
    title: None | str = None,
    **kwargs,
) -> plt.Axes:
    """Plot a 2-d image array from an image stack. If ax is passed, the function will be plotted in
    the provided axis. Additional kwargs are passed to :py:func:`plt.imshow`.

    :param image: 2-d image array from an image stack to plot.
    :param ax: Axes to plot on. If not passed, defaults to the current axes.
    :param title: Title to assign the Axes being plotted on.
    :param kwargs: Additional keyword arguments to pass to plt.imshow.
    :return: The matplotlib Axes containing the plot.

    :example:

        .. code-block:: python

            import numpy as np
            import matplotlib.pyplot as plt
            from scallops.visualize.imshow import imshow_plane

            # Generate a synthetic 2D image array
            image_array = np.random.rand(512, 512)

            # Plot the image using imshow_plane
            fig, ax = plt.subplots()
            imshow_plane(image_array, ax=ax, title="Synthetic Image", cmap="viridis")

            plt.show()
    """
    if ax is None:
        ax = plt.gca()

    if title is not None:
        ax.set_title(title)
    if image.ndim > 2:
        logger.warning("Image is not 2D, sub-setting to the last two dimensions")
        data = image[::-2].squeeze()
    else:
        data = image
    # set imshow default kwargs
    if "cmap" not in kwargs:
        kwargs["cmap"] = plt.cm.gray

    ax.imshow(data, **kwargs)
    ax.axis("off")
    return ax


def plot_plate(
    exp: Experiment,
    plate_shape: tuple[int, int] | Literal["infer"] = (8, 12),
    well_shape: tuple[int, int] = (5, 5),
    channel: int = 0,
    time: int = 0,
    zaxis: int = 0,
    well_layout: int = 9,
    vmax: Literal[""] | int | None = "",
    vmin: Literal[""] | int | None = "",
    full_well: Literal["infer"]
    | Sequence[int]
    | dict[int | str, Sequence[int]]
    | None = None,
    max_workers: int = -1,
    **kwargs,
):
    """Plot raw images per channel in the whole plate layout (mimics heatmap plot).

    :param exp:
        Experiment instance.
    :param plate_shape:
        Tuple with the shape of the plate as (nrows, ncols) or None. If None, only one well is assumed.
        (Default: (8, 12))
    :param well_shape:
        Tuple with the shape of the well as (nrows, ncols). (Default: (5, 5))
    :param channel:
        Channel to plot (default: 0, often is DAPI).
    :param time:
        Time slice to select (default: 0).
    :param zaxis:
        Slice along the Z-axis to take (default: 0).
    :param well_layout:
        Integer representing the order of tile capture (e.g., 9 is snake top to bottom). (Default: 9)
    :param vmax:
        vmax to be passed to all imshow calls.
    :param vmin:
        vmin to be passed to all imshow calls.
    :param full_well:
        Optional sequence of the number of tiles per row in an elliptical well (e.g., Nikon elements tiling).
        If each well has a different shape, a dictionary can be provided, with the well name as the key and the
        sequence of the number of arow as values. Alternatively, 'infer' can be provided if the tile positions are
        available in the image metadata. If not provided (None, the default) is assumed to be a square.
    :param max_workers:
        Number of CPUs to use in inferring coordinates. Only used if full_well is `infer`.
    :param kwargs:
        All additional keyword arguments are passed to the `~matplotlib.pyplot.subplots` call.

    :return:
        The matplotlib Figure containing the plot.
    """
    grouped = groupby(exp.images.keys(), lambda x: x.split("-")[0])
    wells, tiles = zip(*[x.split("-") for x in exp.images.keys()])
    wells = natsorted(set(wells))
    cols, rows = zip(*[(x[0], x[1:]) for x in wells])
    if plate_shape == "infer":
        plate_shape = (len(set(cols)), len(set(rows)))
    if not any(rows):
        rows = range(plate_shape[0])

    if full_well == "infer":
        full_well = _infer_positions(exp, max_workers)
    elif full_well is not None:
        full_well = dict(zip(wells, cycle([full_well])))

    groups = defaultdict(list)
    if full_well is not None:
        for well, tiles in grouped:
            well = set(well)
            assert len(well) == 1, f"Well grouping went wrong: {well}"
            well = next(iter(well))
            idxs = _well_grid(well_shape, tuple(full_well[well]), layout=well_layout)
            si = iter(sorted(zip(idxs[idxs >= 0], tiles), key=lambda x: x[0]))
            flat = chain.from_iterable(
                [tuple(zip(cycle([i]), arr.astype(int))) for i, arr in enumerate(idxs)]
            )
            groups[well].extend(
                [
                    ((r, x), next(si)[1]) if c >= 0 else ((r, x), None)
                    for row, grs in groupby(flat, lambda x: x[0])
                    for x, (r, c) in enumerate(grs)
                ]
            )
    else:
        idxs = _grid_indices(well_layout, well_shape)
        flat = idxs.ravel().astype(tuple)
        groups = {name: tuple(zip(flat, gr)) for name, gr in grouped}

    if plate_shape is not None:
        figsize = np.array(plate_shape)[::-1] * 5
        figsize = tuple(figsize + (figsize * np.array([0.1, 0])))
    elif "figsize" in kwargs:
        figsize = kwargs.pop("figsize")
    else:
        figsize = (5.5, 5)
    for keys in kwargs.keys():
        if keys in ["constrained_layout", "ncols", "nrows"]:
            del kwargs[keys]
    fig, axes = plt.subplots(
        ncols=plate_shape[1] if plate_shape is not None else 1,
        nrows=plate_shape[0] if plate_shape is not None else 1,
        figsize=figsize,
        constrained_layout=True,
        **kwargs,
    )
    axes = np.array(axes)
    axes_idxs = list(zip(*np.where(axes)))
    axes_pd = pd.DataFrame(
        axes, columns=natsorted(set(rows)), index=natsorted(set(cols))
    )
    width, height = 1 / well_shape[0], 1 / well_shape[1]
    if vmin == "" or vmax == "":
        key = choice(list(exp.images))
        vmin, vmax = (
            exp.images[key].isel(c=channel, t=0, z=0).quantile([0.1, 0.9]).values
        )
        vmin -= vmin % +100
        vmax -= vmax % -1000

    def _grid(idx, well_, gr):
        axis = None
        row_name, col_name = well_[0], well_[1:]
        aidx = axes_idxs[idx]
        if not col_name:
            col_name = deepcopy(row_name)
            row_name = aidx[0]
        try:
            ax = axes_pd.loc[row_name, col_name]
        except KeyError as err:
            logger.debug(axes_pd.to_string(), row_name, col_name)
            raise err
        for (arow, acol), tile in gr:
            axins = ax.inset_axes(
                [width * acol, 1 - (height + (height * arow)), width, height]
            )
            axins.set_axis_off()
            if tile is not None:
                try:
                    im = exp.images[tile].isel(t=time, z=zaxis, c=channel)
                    axis = axins.imshow(im, vmax=vmax, vmin=vmin)
                except ValidationError as err:
                    warnings.warn(f"{err}\n skipping tile {tile}")
                except IndexError as err:
                    logger.debug(tile, time, zaxis, channel)
                    raise err
        if (np.where(axes_pd.values == ax)[1] == 0) and not isinstance(row_name, int):
            ax.set_ylabel(
                row_name, rotation="horizontal", fontdict={"fontsize": 20}, labelpad=8.0
            )
        if aidx[0] == 0:
            ax.set_title(col_name, fontdict={"fontsize": 20})
        return axis

    ims = [_grid(i, name, gr) for i, (name, gr) in enumerate(groups.items())]
    for ax in axes.flat:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        for key_, spine in ax.spines.items():
            spine.set_visible(False)
    ims = [x for x in ims if x is not None]
    # (left, bottom, width, height)
    cbar_ax = fig.add_axes([1.05, 0.05, 0.05, 0.9])
    _ = fig.colorbar(ims[-1], cax=cbar_ax)
    return fig


def tiles_over_stitch(
    img: str | DataArray, stitched_coords: str | pd.DataFrame, img_channel=0
):
    """Display an image with overlaid tiles.

    :param img: Image file path or xarray DataArray containing the image data.
    :param stitched_coords: CSV file path or pandas DataFrame with stitched coordinates.
    :param img_channel: Channel index for the image, defaults to 0.
    """
    if isinstance(img, str):
        img = read_image(img)
    img = img.isel(c=img_channel).squeeze()
    if isinstance(stitched_coords, str):
        stitched_coords = pd.read_csv(stitched_coords)
    f, ax = plt.subplots(figsize=(10, 10), layout="constrained")
    ax.imshow(img, vmax=np.quantile(img, 0.85))
    ax.axis("off")
    for tup in stitched_coords.itertuples():
        bbox_width, bbox_height = tup.XSize, tup.YSize
        x, y = tup.X, tup.Y
        rect = patches.Rectangle(
            (x, y),
            bbox_width,
            bbox_height,
            edgecolor=None,
            facecolor="white",
            alpha=0.8,
            zorder=1,
        )  # Ensure the rectangle is behind the text
        ax.add_patch(rect)
        ax.text(
            x + ((bbox_width - 13.60) / 2),
            y + ((bbox_height - 13.60) / 2),
            tup.Tile,
            fontsize=8,
        )
    return f, ax


def plot_percentile_montage(
    df: pd.DataFrame,
    genes_to_plot: Sequence[str],
    images: str | xr.DataArray | Experiment,
    labels: str | xr.DataArray | Experiment | None = None,
    feature: str = "nuclei_mean_7",
    group_by: Sequence[str] = ("Plate", "Row", "gene_symbol"),
    percentiles: Sequence[float] = (10, 30, 50, 70, 90),
    pad: int = 50,
    guide_lines: bool = False,
    context: bool = False,
    gene_name_col: str = "gene_symbol",
    cmap: str | None = None,
    vmaxq: float = 0.9,
    vminq: float = 0.1,
    labels_key_pattern: str | None = None,
) -> tuple[plt.Figure, dict[str, list[tuple[int, str, int, int]]]]:
    """Plot a montage of gene-specific images at specified feature percentiles from a dataframe and
    image stack.

    This function creates a montage of images for specified genes, displaying representative cells at given percentiles
    of a specified feature. It allows for context-based padding and optional guide-lines, extracting relevant image
    segments and aligning them based on their maximum size.

    :param df: DataFrame containing metadata with positional information for each gene.
    :param genes_to_plot: Sequence of gene symbols to plot.
    :param images: Image data, can be a string path, xarray DataArray, or Experiment object.
    :param labels: Label data from segmentation if context is disabled. Provide either labels or set context to True,
        not both. Defaults to None.
    :param feature: Feature column name to use for percentile calculations. Defaults to 'nuclei_mean_7'.
    :param group_by: Column names to group by when calculating percentiles. Defaults to ('Plate', 'Row', 'gene_symbol').
    :param percentiles: Percentiles to select representative cells. Defaults to (10, 30, 50, 70, 90).
    :param pad: Padding size to apply around each extracted image. Defaults to 50.
    :param guide_lines: Whether to include guide-lines on the montage. Defaults to False.
    :param context: If True, uses the entire image region without segmentation labels. Provide either labels or set
        context to True, not both. Defaults to False.
    :param gene_name_col: Column name in the dataframe that contains gene names. Defaults to 'gene_symbol'.
    :param cmap: Colormap to use for displaying images. Defaults to None.
    :param vmaxq: Upper quantile for color scaling. Defaults to 0.9.
    :param vminq: Lower quantile for color scaling. Defaults to 0.1.
    :param labels_key_pattern: String pattern of the type '{well}-nuclei' to index labels if using Experiment.
        Defaults to None.
    :return: A tuple containing the matplotlib Figure object and a dictionary mapping gene symbols to selected
        coordinates.


    :example:

    .. code-block:: python

        import pandas as pd
        import xarray as xr
        import matplotlib.pyplot as plt
        from scallops.visualize import plot_percentile_montage

        # Example usage of plot_percentile_montage function
        df = pd.read_csv("data.csv")
        genes_to_plot = ["GeneA", "GeneB"]
        images = xr.load_dataarray("images.nc")  # Load images as xarray DataArray
        labels = xr.load_dataarray(
            "labels.nc"
        )  # Optional: Load labels as xarray DataArray

        fig, selected = plot_percentile_montage(
            df,
            genes_to_plot,
            images,
            labels=labels,
            feature="nuclei_mean_7",
            guide_lines=True,
            context=False,
        )
        plt.show()
    """
    desired_order = ["label", "nuclei_centroid-0", "nuclei_centroid-1"]
    # Check that the required feature and columns are present
    assert feature in df.columns, f'Feature "{feature}" not in dataframe.'
    assert isinstance(group_by, Sequence), "`group_by` must be a sequence"
    if isinstance(group_by, str):
        group_by = [group_by]
    else:
        group_by = list(group_by)

    # Check for images and labels types
    single_well = True
    if isinstance(images, str):
        # Assumes it is a well for simplicity. Only Experiment accepts plate
        images = read_image(images, dask=True)
    elif isinstance(images, Experiment):
        # Since it might be a plate, check that the well is present
        assert "well" in df.columns, (
            "Column 'well' must be present when using Experiment"
        )
        desired_order.append("well")
        single_well = False
    else:
        assert isinstance(images, xr.DataArray), (
            "Images can only be Experiment or xarrays"
        )

    if labels is None:
        assert context, "Either provide labels or set context to True"
    elif isinstance(labels, str):
        # Assumes it is a well for simplicity. Only Experiment accepts plate
        labels = read_image(labels, dask=True)
    if labels is not None:
        assert not context, "Provide either labels or set context to True, not both"

    desired_order.extend([gene_name_col, feature])
    required_cols = set(group_by + desired_order)
    missing_cols = required_cols - set(df.columns)
    assert not missing_cols, f"Missing columns in dataframe: {', '.join(missing_cols)}"
    remaining_columns = list(required_cols - set(desired_order))
    required_cols = desired_order + remaining_columns

    # Filter and prepare the dataframe
    df = df.dropna(subset=[feature, gene_name_col])[required_cols]
    missing_genes = set(genes_to_plot) - set(df[gene_name_col])
    assert not missing_genes, f"Some genes are not in the dataframe: {missing_genes}"
    df = df[df[gene_name_col].isin(genes_to_plot)]

    # tokenize feature
    _, _, ch, *_ = feature.split("_")
    ch = int(ch)
    # if array, make sure the channel is there or assume is first
    if isinstance(images, xr.DataArray):
        if ch > images.c.size:
            assert images.c.size == 1, (
                f"Image channel  {ch} larger than image channels, and multiple channel passed"
            )
            ch = 0

    # Get representative rows at specified percentiles
    def get_representative_rows(group):
        percentile_values = np.percentile(group[feature], percentiles)
        differences = abs(group[feature].values[:, np.newaxis] - percentile_values)
        idx_min = differences.argmin(axis=0)
        representative_rows = group.iloc[idx_min].copy()
        representative_rows["percentile"] = percentiles
        return representative_rows

    # Apply the function to get representative rows
    df = (
        df.groupby(group_by)
        .apply(get_representative_rows, include_groups=False)
        .reset_index(level=group_by)
        .reset_index(drop=True)
        .sort_values(by=group_by + ["percentile"])
    )[required_cols + ["percentile"]]

    ngenes = len(genes_to_plot)
    ncols = len(percentiles)
    f, axs = plt.subplots(
        nrows=ngenes, ncols=ncols, figsize=(5 * ncols, 5 * ngenes), layout="constrained"
    )

    def get_slices(row):
        if single_well:
            img = images
            lab = labels
        else:
            well = row.well
            img = images.images[f"{well}"]
            lab = labels.labels[labels_key_pattern.format(well=well)]
        x = int(round(row[2]))
        y = int(round(row[3]))
        xslice = slice(max(0, y - pad), min(y + pad, img.sizes["x"]))
        yslice = slice(max(0, x - pad), min(x + pad, img.sizes["y"]))
        img = img.isel(c=ch, x=xslice, y=yslice)
        if not context and labels is not None:
            lab = lab.isel(x=xslice, y=yslice)
            props = regionprops_table(
                lab.values.astype(int),
                intensity_image=img.values,
                properties=("label", "image_intensity"),
            )
            props_df = pd.DataFrame(props)
            im = xr.DataArray(
                props_df.query(f"label == {row.label}")["image_intensity"].values[0],
                dims=["y", "x"],
            )
        else:
            im = img.copy()
        max_size = max(im.sizes["x"], im.sizes["y"], pad)
        pad_x = max(0, max_size - im.sizes["x"])
        pad_y = max(0, max_size - im.sizes["y"])
        im = im.pad({"x": (0, pad_x), "y": (0, pad_y)}, constant_values=0)
        return resize(im.values, (pad, pad))

    cols = {
        gene: [
            get_slices(row)
            for row in df.query(f"{gene_name_col} == '{gene}'").itertuples()
        ]
        for gene in genes_to_plot
    }

    # Determine color scaling
    vmin, vmax = np.nanquantile(
        np.concatenate([np.array(imgs) for imgs in cols.values()]), [vminq, vmaxq]
    )

    # Plot the images
    axs = np.atleast_2d(axs)  # Ensure axs is 2D
    for i, gene in enumerate(genes_to_plot):
        d_gene = df[df[gene_name_col] == gene]
        for j, (ax, perc) in enumerate(zip(axs[i], d_gene["percentile"])):
            img = cols[gene][j]
            try:
                ax.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)
            except Exception as e:
                print(f'Error plotting gene "{gene}": {e}')
                raise
            # Remove ticks and spines but keep labels
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == ngenes - 1:
                ax.set_xlabel(f"{perc}%", fontsize=12)
            if j == 0:
                ax.set_ylabel(f"{gene}", fontsize=12)
            if guide_lines:
                ax.axvline(pad / 2, color="white", linestyle=":", alpha=0.8)
                ax.axhline(pad / 2, color="white", linestyle=":", alpha=0.8)

    return f, cols
