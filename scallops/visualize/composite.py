"""Composite Image Visualization Submodule.

Provides functions for generating composite images from multi-dimensional imaging data.

Authors:
- The SCALLOPS development team
"""

from collections.abc import Sequence
from copy import copy
from typing import Any

import dask.array as da
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap, ListedColormap
from scipy import ndimage as ndi

from scallops.experiment.elements import Experiment
from scallops.experiment.util import _concat_images
from scallops.visualize.utils import (
    _create_color_map_for_rgb,
    _figsize,
    _get_image_crop_slice,
    _wrap_cols,
    channel_thresholds,
)
from scallops.xr import _crop


def _create_labels_cmap(
    labels_cmap: str | ListedColormap | None, labels_alpha: float
) -> ListedColormap:
    """Create a color map for labeling with an adjustable alpha channel.

    :param labels_cmap: Name of the color map, a ListedColormap instance, or None.
    :param labels_alpha: Alpha value for the labels' color map.
    :return: A ListedColormap with adjusted alpha values.
    """
    if isinstance(labels_cmap, str):
        labels_cmap = plt.get_cmap(labels_cmap)
    if isinstance(labels_cmap, ListedColormap):
        label_colors = labels_cmap.colors
        if len(label_colors[0]) == 3:
            label_colors = [c + (labels_alpha,) for c in label_colors]
        else:
            label_colors = [c[:3] + (labels_alpha,) for c in label_colors]
        return ListedColormap(label_colors)
    if labels_cmap is None:
        return random_labels_cmap(labels_alpha)
    raise ValueError("Unable to create label color map")


def random_labels_cmap(
    alpha: float, n: int = 2**16, h=(0, 1), light=(0.4, 1), s=(0.2, 0.8)
) -> ListedColormap:
    """Generate a random color map for labels with adjustable alpha.

    :param alpha: Alpha value for the color map.
    :param n: Number of colors to generate.
    :param h: Hue range.
    :param light: Lightness range.
    :param s: Saturation range.
    :return: A ListedColormap with random colors.
    """
    import colorsys

    seed = np.random.seed()
    np.random.seed(0)
    h, light, s = (
        np.random.uniform(*h, n),
        np.random.uniform(*light, n),
        np.random.uniform(*s, n),
    )
    np.random.seed(seed)
    cols = np.stack(
        [colorsys.hls_to_rgb(_h, _l, _s) for _h, _l, _s in zip(h, light, s)], axis=0
    )
    _alpha = np.full((len(cols), 1), alpha)
    cols = np.hstack([cols, _alpha])
    return ListedColormap(cols)


def _get_vmin_max(d, j, sel):
    """Get the minimum and maximum values for normalization.

    :param d: Data input (can be DataArray or numeric).
    :param j: Index for selection.
    :param sel: Selection dictionary.
    :return: The selected value or data slice.
    """

    if isinstance(d, xr.DataArray):
        d = d.isel(sel).data if d.ndim > 0 else d.data
    elif d is not None:
        try:
            d = d[j]
            if isinstance(d, xr.DataArray):
                d = d.data
        except:  # noqa: E722
            pass
    return d


def _imcomposite_image(
    image: xr.DataArray | np.ndarray | da.Array | None = None,
    vmin: float | xr.DataArray | np.ndarray | da.Array | None = None,
    vmax: float | xr.DataArray | np.ndarray | da.Array | None = None,
    cmap: None | list[str | Colormap] | str | Colormap = None,
    dim: None | str = "c",
    rgb: bool = False,
):
    """Create a composite image from input data with optional colormaps.

    :param image: Input image data.
    :param vmin: Minimum normalization value.
    :param vmax: Maximum normalization value.
    :param cmap: Colormap or list of colormaps.
    :param dim: Dimension to process.
    :param rgb: Whether the input is an RGB image.
    :return: A composite image.
    """

    dim_size = 0

    if image is not None and not rgb:
        image = image.squeeze()
        if not isinstance(image, xr.DataArray) or (
            dim is not None and dim not in image.dims
        ):
            dim = None
        dim_size = image.sizes[dim] if dim is not None else 1
        if dim is None and image.ndim != 2:
            raise ValueError("Expecting image to have 2 dimensions")
        if dim is not None and image.ndim != 3:
            raise ValueError("Expecting image to have 3 dimensions")

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    dim_index_to_cmap = {}

    # colormap : (red, blue) for 2 channels, (CYMRGB) for more than 2
    # napari default of ['cyan', 'yellow', 'magenta', 'red', 'green', 'blue']
    CYMRGB = [[0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    RED_BLUE = [[1, 0, 0], [0, 0, 1]]
    rgbs = RED_BLUE if dim_size <= 2 else CYMRGB

    for j in range(dim_size):
        cmap_j = cmap[j % len(cmap)] if isinstance(cmap, Sequence) else cmap
        if cmap_j is None:
            cmap_j = _create_color_map_for_rgb(rgbs[j % len(rgbs)])
        if isinstance(cmap_j, str):
            cmap_j = plt.get_cmap(cmap_j)
        dim_index_to_cmap[j] = cmap_j

    im_combined = None
    if image is not None and not rgb:
        for j in range(dim_size):
            sel = dict()
            if dim is not None:
                sel[dim] = j
            image_values = (
                image.isel(sel).data if isinstance(image, xr.DataArray) else image
            )
            assert image_values.ndim == 2
            if im_combined is None:
                im_combined = np.zeros(image_values.shape + (4,))

            vmin_j = _get_vmin_max(vmin, j, sel)
            if vmin_j is None:
                vmin_j = image_values.min()
            vmax_j = _get_vmin_max(vmax, j, sel)
            if vmax_j is None:
                vmax_j = image_values.max()
            denom = vmax_j - vmin_j
            if denom == 0:
                denom = 1
            image_values = (image_values - vmin_j) / denom
            image_values[image_values > 1] = 1
            image_values[image_values < 0] = 0

            channel_cmap = dim_index_to_cmap[j]
            im_combined += channel_cmap(image_values)
        im_combined[im_combined > 1] = 1
        im_combined = im_combined.squeeze()
    elif image is not None and rgb:
        im_combined = image
    return im_combined


def _imcomposite_labels(
    labels: xr.DataArray | np.ndarray | None = None,
    labels_cmap: str | ListedColormap | None = None,
    labels_alpha: float = 1,
):
    """Create a composite image for labels.

    :param labels: Input label data.
    :param labels_cmap: Colormap for the labels.
    :param labels_alpha: Alpha value for the labels.
    :return: A tuple of the composite image and label mask.
    """
    cmap = _create_labels_cmap(labels_cmap, labels_alpha)
    im_lbl = cmap(((labels - 1) % len(cmap.colors)).astype(int))
    return im_lbl, labels > 0


def _get_contours(
    labels: np.ndarray | xr.DataArray, thickness: int, background_label: int
):
    """From napari.napari/layers/labels/_labels_utils.py"""
    if isinstance(labels, xr.DataArray):
        labels = labels.data
    struct_elem = ndi.generate_binary_structure(labels.ndim, 1)

    thick_struct_elem = ndi.iterate_structure(struct_elem, thickness).astype(bool)

    dilated_labels = ndi.grey_dilation(labels, footprint=struct_elem)
    eroded_labels = ndi.grey_erosion(labels, footprint=thick_struct_elem)
    not_boundaries = dilated_labels == eroded_labels

    contours = labels.copy()
    contours[not_boundaries] = background_label

    return contours


def _imcomposite_image_labels(
    image=None,
    labels=None,
    vmin=None,
    vmax=None,
    cmap=None,
    labels_cmap=None,
    labels_alpha=0.5,
    labels_contour=False,
    labels_contour_thickness=1,
    labels_contour_cmap=None,
    labels_contour_alpha=1.0,
    dim="c",
    rgb=False,
):
    """Create a composite image with labels and optional contours.

    :param image: Input image data.
    :param labels: Label data.
    :param vmin: Minimum normalization value.
    :param vmax: Maximum normalization value.
    :param cmap: Colormap(s) for the image.
    :param labels_cmap: Colormap for the labels.
    :param labels_alpha: Alpha value for the labels.
    :param labels_contour: Whether to display label contours.
    :param labels_contour_cmap: Colormap for the contours.
    :param dim: Dimension to process.
    :param rgb: Whether the input is an RGB image.
    :return: A composite image with labels.
    """
    im_combined = _imcomposite_image(
        image=image, vmin=vmin, vmax=vmax, cmap=cmap, dim=dim, rgb=rgb
    )
    if isinstance(labels_contour, bool) and labels_contour and labels is not None:
        labels_contour = labels
    if labels is not None:
        im_lbl, mask_lbl = _imcomposite_labels(labels=labels, labels_cmap=labels_cmap)

        if im_combined is None:
            im_combined = np.zeros_like(im_lbl)

        # blend image and label
        im_combined[mask_lbl] = (
            labels_alpha * im_lbl[mask_lbl] + (1 - labels_alpha) * im_combined[mask_lbl]
        )

    # contour
    if isinstance(labels_contour, (np.ndarray, da.Array, xr.DataArray)):
        # use a different mask for contours
        if labels_contour_cmap is not None:
            labels_cmap = labels_contour_cmap

        labels_contour = _get_contours(labels_contour, labels_contour_thickness, 0)
        im_lbl, mask_lbl = _imcomposite_labels(
            labels=labels_contour, labels_cmap=labels_cmap
        )

        if im_combined is None:
            im_combined = np.zeros_like(im_lbl)
        im_combined[mask_lbl] = (
            labels_contour_alpha * im_lbl[mask_lbl]
            + (1 - labels_contour_alpha) * im_combined[mask_lbl]
        )

    return im_combined


def imcomposite(
    image: xr.DataArray | np.ndarray | da.Array | None = None,
    labels: xr.DataArray | np.ndarray | da.Array | None = None,
    vmin: float | xr.DataArray | np.ndarray | da.Array | None = None,
    vmax: float | xr.DataArray | np.ndarray | da.Array | None = None,
    cmap: Sequence[str | Colormap] | str | Colormap | None = None,
    labels_cmap: str | ListedColormap | None = None,
    labels_alpha: float = 0.5,
    labels_contour: bool | np.ndarray | xr.DataArray | da.Array = False,
    labels_contour_cmap: str | ListedColormap | None = None,
    labels_contour_thickness: int = 1,
    labels_contour_alpha: float = 1,
    dim: str | None = "c",
    figsize: tuple[int, int] = None,
    ax: plt.Axes | None = None,
    rgb: bool = False,
    mask: np.ndarray | xr.DataArray | da.Array = None,
) -> plt.Axes:
    """Plot an image composite using additive blending.

    :param image: XArray with dimensions y, x and optionally `dim`.
    :param labels: Labels (e.g., segmentation labels).
    :param vmin: Lower color limit used for determining the colormap bounds. Default
        is the minimum of the image. If a list, must be the same length as the
        dimension being expanded.
    :param vmax: Upper color limit used for determining the colormap bounds. Default
        is the maximum of the image. If a list, must be the same length as the
        dimension being expanded.
    :param figsize: Figure size.
    :param cmap: List that maps dimension index to colormap instance or registered
        colormap name used to map scalar data to colors.
    :param labels_cmap: ListedColormap instance or registered colormap name used to
        map label data to colors.
    :param labels_alpha: Alpha value for labels.
    :param labels_contour: If true, show label contours only. If an array, then
        segmentation labels to show as contour.
    :param labels_contour_cmap: ListedColormap instance or registered colormap name
        used to map label contour data to colors.
    :param labels_contour_thickness: Labels contour thickness.
    :param labels_contour_alpha: Alpha value for label contours.
    :param labels_cmap: ListedColormap instance or registered colormap name used to
        map label data to colors.
    :param dim: Image dimension to blend.
    :param ax: Matplotlib axes to plot the composite to.
    :param rgb: Whether the image is RGB or RGBA.
    :param mask: Mask pixels where mask == 0

    :example:

        .. code-block:: python

            from matplotlib import pyplot as plt
            import numpy as np
            import xarray as xr
            from scallops.visualize import imcomposite

            # Generate synthetic data for testing
            t, c, z, y, x = (
                1,
                3,
                1,
                10,
                10,
            )  # emulate all possible dimensions (only 3D is accepted by single)
            image_data = np.random.rand(t, c, z, y, x)
            labels_data = np.random.randint(0, 2, size=(y, x))

            image = xr.DataArray(image_data, dims=("t", "c", "z", "y", "x"))
            labels = xr.DataArray(labels_data, dims=("y", "x"))

            # Plot the composite image
            ax = imcomposite(image.isel(t=0, z=0), labels, figsize=(8, 8), dim="c")
            plt.show()
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize if figsize is not None else (6, 6))

    ax.axis("off")
    im_combined = _imcomposite_image_labels(
        image=image,
        labels=labels,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        labels_cmap=labels_cmap,
        labels_alpha=labels_alpha,
        labels_contour=labels_contour,
        labels_contour_thickness=labels_contour_thickness,
        labels_contour_alpha=labels_contour_alpha,
        labels_contour_cmap=labels_contour_cmap,
        dim=dim,
        rgb=rgb,
    )
    if mask is not None:
        mask_alpha = 0.5
        mask_img = _imcomposite_image(
            mask, dim=None, cmap=ListedColormap([(0, 0, 0, mask_alpha)])
        )
        mask_sel = mask == 0
        im_combined[mask_sel] = (
            mask_alpha * mask_img[mask_sel] + (1 - mask_alpha) * im_combined[mask_sel]
        )
    if im_combined is not None:
        ax.imshow(im_combined, cmap=None)
    return ax


def experiment_composite(
    exp: Experiment,
    subset: list[str] | None = None,
    max_cols: int = 3,
    figsize: tuple[int, int] = (5, 5),
    **kwargs,
) -> tuple[plt.Figure, plt.Axes | np.ndarray[plt.Axes]]:
    """Plot an experiment (one image per axis) composite using additive blending.

    :param figsize: Figure size
    :param exp: Scallop experiment to plot
    :param subset: List of image keys to plot, by default all present in `experiment`
    :param max_cols: Maximum number of plotting columns
    :param kwargs: Key-word arguments passed to `single` plotting function
    """
    image_keys = exp.images.keys() if subset is None else subset
    ncols = min(max_cols, len(image_keys))
    nrows = np.round(len(image_keys) / ncols).astype(int)
    f, axs = plt.subplots(
        nrows=nrows, ncols=ncols, layout="constrained", figsize=figsize
    )
    for ax in axs.flat:
        ax.set_axis_off()
    for im, ax in zip(image_keys, axs.flat):
        ind_kwargs = copy(kwargs)
        if "labels" in ind_kwargs:
            ind_kwargs["labels"] = ind_kwargs["labels"].labels[im]
        img = exp.images[im].isel(t=0, z=0)
        ax.set_axis_on()
        ax.set_title(im)
        imcomposite(image=img, ax=ax, **ind_kwargs)
    return f, axs


def label_montage(
    image: xr.DataArray,
    labels: np.ndarray,
    labels_include: list[int],
    col_wrap: int = 14,
    crop_size: tuple[int, int] = (24, 24),
    **kwargs: Any,
) -> np.ndarray:
    """Plot a label montage.

    :param image: XArray representing image to plot
    :param labels: Labels (e.g., segmentation labels).
    :param labels_include: list of label ids to plot
    :param col_wrap: Number of columns in grid
    :param crop_size: Size of label crop
    :param kwargs: Keyword arguments passed to `single` imcomposite function (e.g. vmin)
    :return: RGBA array

    :example:

        .. code-block:: python

            import numpy as np
            import xarray as xr
            from matplotlib import pyplot as plt
            from matplotlib.colors import ListedColormap
            from skimage import data
            from skimage.filters import sobel
            from skimage.measure import label
            from skimage.segmentation import expand_labels, watershed

            from scallops.visualize.composite import label_montage

            coins = data.coins()

            # Make segmentation using edge-detection and watershed.
            edges = sobel(coins)

            # Identify some background and foreground pixels from the intensity values.
            # These pixels are used as seeds for watershed.
            markers = np.zeros_like(coins)
            foreground, background = 1, 2
            markers[coins < 30.0] = background
            markers[coins > 150.0] = foreground

            ws = watershed(edges, markers)
            seg1 = label(ws == foreground)

            expanded = expand_labels(seg1, distance=10)
            img = label_montage(
                image=xr.DataArray(coins, dims=["y", "x"]),
                labels=expanded,
                labels_include=np.unique(expanded)[1:18],
                crop_size=(80, 80),
                col_wrap=6,
                labels_contour_cmap=ListedColormap([(1, 0, 0, 1)]),
            )
            plt.imshow(img, cmap=None)
            plt.axis("off")
    """

    ncol, nrow = _wrap_cols(ncol=len(labels_include), col_wrap=col_wrap)
    outline = np.zeros(labels.shape)

    im_combined = np.zeros((crop_size[0] * nrow, crop_size[1] * ncol) + (4,))
    col_index = 0
    row_index = 0
    imcomposite_args = dict(labels_contour_cmap=ListedColormap([(1, 1, 1, 1)]))
    imcomposite_args.update(**kwargs)
    for label in labels_include:
        outline[...] = 0
        indices = np.where(labels == label)
        outline[indices[0], indices[1]] = 1
        y = _get_image_crop_slice(indices[0], image.sizes["y"], crop_size[0], False)
        x = _get_image_crop_slice(indices[1], image.sizes["x"], crop_size[1], False)
        img = _imcomposite_image_labels(
            image.isel(y=y, x=x), labels_contour=outline[y, x], **imcomposite_args
        )

        ypix = row_index * crop_size[0]
        xpix = col_index * crop_size[1]
        im_combined[ypix : ypix + crop_size[0], xpix : xpix + crop_size[1]] = img
        col_index = col_index + 1
        if col_index == ncol:
            col_index = 0
            row_index = row_index + 1
    return im_combined


def montage_plot(
    image: xr.DataArray | Experiment,
    percentile_min: Sequence[float] | float = 0.0,
    percentile_max: Sequence[float] | float = 99.0,
    pad_min: Sequence[float] | float = 100,
    pad_max: Sequence[float] | float = 3000,
    figsize: tuple[int, int] | None = None,
    thresholds: dict[int, tuple[float, float]] = None,
    row_labels: str | Sequence[str] | None = None,
    col_labels: str | Sequence[str] | None = None,
    crop: int | tuple[int, int, int, int] | None = None,
    display_t: bool | None = False,
    cmap: None | Sequence[str, str | Colormap] | str | Colormap = None,
):
    """Plot an image montage, which is a composite view of multiple images arranged
    in a grid.

    A montage provides a compact visualization of multiple images, often used in
    scientific and medical imaging to compare variations or features across different
    samples or channels.

    :param image: XArray with dimensions (t, c, z, y, x) or (i, t, c, z, y, x).
    :param percentile_min: Lower percentile used to calculate contrasting thresholds.
        If a list, different percentiles per channel.Default is 0.0. If a number, the
        same percentile is applied for all channels.
    :param percentile_max: Upper percentile used to calculate contrasting thresholds.
        If a list, different percentiles per channel. Default is 99.0. If a number,
        the same percentile is applied for all channels.
    :param pad_min: vmin = tmin - pad_min and vmax = tmax + pad_max. If vectors,
        different pads per channel. Otherwise, apply the same pad.
    :param pad_max: vmax = tmax + pad_max. If vectors, different pads per channel.
        Otherwise, apply the same pad.
    :param figsize:  Figure size.
    :param thresholds:  Optional dictionary that maps channel to a tuple of
        (tmin, tmax) instead of computing thresholds using percentiles and pad.
    :param row_labels: Image attribute to show along rows (e.g., well_row) or a list of
        labels.
    :param col_labels: Image attribute to show along columns (e.g., well_row) or a
        list of labels.
    :param crop: Pixel size for cropping each panel at the center (e.g., crop=300
        means only show 300x300 at the center) or tuple of (x, y, width, height).
    :param display_t: Use the T dimension in rows
    :param cmap: Sequence of colormaps or registered colormap name used to map scalar
        data to colors.

    :example:

        .. code-block:: python

            # Generate synthetic data for testing
            import numpy as np
            import xarray as xr
            from scallops.visualize import montage_plot

            image_data = np.random.rand(t, c, z, y, x)
            image = xr.DataArray(image_data, dims=("t", "c", "z", "y", "x"))

            # Plot the image montage
            montage_plot(
                image.isel(t=0, z=0),
                percentile_min=0,
                percentile_max=1,
                pad_min=0.01,
                pad_max=0.99,
                figsize=(12, 8),
            )
            plt.show()
    """

    if isinstance(image, Experiment):
        image = _concat_images(image)
    if display_t:
        row_labels = image.t.values if row_labels is None else row_labels
        image = image.rename({"t": "i"})
    if "t" in image.sizes:
        image = image.isel(t=0)
    channels = image.c.values
    nchannels = len(channels)
    if crop is not None:
        image = _crop(image, crop)
    _thresholds = channel_thresholds(
        image=image,
        percentile_min=percentile_min,
        percentile_max=percentile_max,
        pad_min=pad_min,
        pad_max=pad_max,
        thresholds=thresholds,
    )
    # [black, green, red, magenta, cyan, yellow, blue]
    default_rgbs = [
        [1, 1, 1],
        [0, 1, 0],
        [1, 0, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
        [0, 0, 1],
    ]
    nimages = image.sizes["i"] if "i" in image.dims else 1
    figsize = _figsize(ncol=nchannels, nrow=nimages) if figsize is None else figsize
    fig, axes = plt.subplots(nimages, nchannels, figsize=figsize, squeeze=False)

    for i in range(nimages):
        data_array = image.isel(i=i, missing_dims="ignore")
        for j in range(nchannels):
            ax = axes[i, j]
            c = channels[j]
            if cmap is None:
                cmap_j = _create_color_map_for_rgb(default_rgbs[j % len(default_rgbs)])
            else:
                cmap_j = cmap[j % len(cmap)] if isinstance(cmap, Sequence) else cmap
            if i == 0:
                if col_labels is None:
                    title = str(c)
                    if thresholds and c in thresholds:
                        vmin, vmax = _thresholds[c]
                        title = f"{title} ({vmin:.1f}-{vmax:.1f})"
                else:
                    title = (
                        col_labels[j]
                        if not isinstance(col_labels, str)
                        else data_array.attrs[col_labels]
                    )
                ax.set_title(title)
            ax.axis("off")
            data = data_array.sel(c=c).squeeze().values
            vmin, vmax = _thresholds[c]
            ax.imshow(data, cmap=cmap_j, vmin=vmin, vmax=vmax)
        if row_labels is not None:
            left_axis = axes[i, 0]
            left_axis.axis("on")
            left_axis.get_yaxis().set_ticks([])
            left_axis.get_xaxis().set_visible(False)
            y_label = (
                row_labels[i]
                if not isinstance(row_labels, str)
                else data_array.attrs[row_labels]
            )
            left_axis.set_ylabel(y_label)
    fig.tight_layout()
