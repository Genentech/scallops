import dask.array as da
import ipywidgets as widgets
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt

from scallops.io import get_image_spacing
from scallops.stitch.utils import min_zncc
from scallops.visualize import imcomposite


def browse_labels(
    image1: xr.DataArray,
    image2: xr.DataArray | None,
    labels_df: pd.DataFrame,
    labels: xr.DataArray | None = None,
    padding: float | None = 25,
    figsize: tuple[int, int] = (10, 10),
) -> widgets.Widget:
    """Create a Jupyter widget allows users to select labels of interest.

    :param image1: Image one to browse
    :param image2: Optional image two to browse
    :param labels_df: Labels dataframe with labels as index, and ordered columns
        containing AreaShape_BoundingBoxMinimum_Y,
        AreaShape_BoundingBoxMinimum_X, AreaShape_BoundingBoxMaximum_Y,
        AreaShape_BoundingBoxMaximum_X
    :param labels: Labels array
    :param padding: Padding in microns to add to each label
    :param figsize: Figure size
    :return: Jupyter widget
    """
    return _browse_regions_or_labels(
        image1=image1,
        image2=image2,
        labels_df=labels_df,
        figsize=figsize,
        padding=padding,
        labels=labels,
    )


def browse_image_regions(
    image1: xr.DataArray,
    image2: xr.DataArray | None,
    y_indices: np.ndarray[int],
    x_indices: np.ndarray[int],
    labels: xr.DataArray | None = None,
    figsize: tuple[int, int] = (10, 10),
) -> widgets.Widget:
    """Create a Jupyter widget allows users to select regions of interest.

    :param image1: Image one to browse
    :param image2: Optional image two to browse
    :param y_indices: Indices of y coordinates
    :param x_indices: Indices of x coordinates
    :param labels: Labels array
    :param figsize: Figure size
    :return: Jupyter widget
    """
    return _browse_regions_or_labels(
        image1=image1,
        image2=image2,
        y_indices=y_indices,
        x_indices=x_indices,
        figsize=figsize,
        labels=labels,
    )


def _browse_regions_or_labels(
    image1: xr.DataArray,
    image2: xr.DataArray | None = None,
    labels: xr.DataArray | None = None,
    y_indices: np.ndarray[int] | None = None,
    x_indices: np.ndarray[int] | None = None,
    labels_df: pd.DataFrame | None = None,
    padding: float | None = None,
    figsize: tuple[int, int] = (10, 10),
) -> widgets.Widget:
    if image2 is not None:
        assert image1.shape == image2.shape
    image_spacing = get_image_spacing(image1.attrs)
    if padding is not None:
        padding = np.array([padding, padding]) / image_spacing
        padding = padding.astype(int)
    chunk_out = widgets.Output()
    if y_indices is not None:
        assert x_indices is not None
        img_slice_size = image1.data.chunksize[-2:]

        selector = widgets.Dropdown(
            options=[
                (
                    f"{y_indices[i] * img_slice_size[0] * image_spacing[0]:,.1f}, "
                    f"{x_indices[i] * img_slice_size[1] * image_spacing[1]:,.1f}",
                    i,
                )
                for i in range(len(y_indices))
            ],
            value=0 if len(y_indices) > 0 else None,
            description="Region:",
            disabled=len(y_indices) == 0,
        )
    else:
        assert labels_df is not None
        selector = widgets.Dropdown(
            options=[
                (f"{labels_df.index.values[i]}", i) for i in range(len(labels_df))
            ],
            value=0 if len(labels_df) > 0 else None,
            description="Label:",
            disabled=len(labels_df) == 0,
        )

    def _on_index_changed(event):
        with chunk_out:
            chunk_out.clear_output()
            index = event.new if event is not None else 0

            if y_indices is not None:
                y_px = y_indices[index] * img_slice_size[0]
                x_px = x_indices[index] * img_slice_size[1]
                y = slice(y_px, y_px + img_slice_size[0])
                x = slice(x_px, x_px + img_slice_size[1])

            else:
                sel = labels_df.iloc[[index]]
                y1 = sel[labels_df.columns[0]].values[0] - padding[0]
                x1 = sel[labels_df.columns[1]].values[0] - padding[1]
                y2 = sel[labels_df.columns[2]].values[0] + padding[0]
                x2 = sel[labels_df.columns[3]].values[0] + padding[1]
                y = slice(
                    y1,
                    y2,
                )
                x = slice(
                    x1,
                    x2,
                )

            a = image1.isel(y=y, x=x)
            b = image2.isel(y=y, x=x) if image2 is not None else None

            dim = "i"
            if b is None:
                dim = "t" if "t" in image1.dims and image1.sizes["t"] > 1 else "c"
            labels_ = None
            if labels is not None:
                labels_ = labels.isel(y=y, x=x)
                if labels_df is not None:
                    selected_label = sel.index.values[0]
                    labels_ = labels_.data
                    if isinstance(labels_, da.Array):
                        labels_ = labels_.compute()
                    else:
                        labels_ = labels_.copy()
                    labels_[labels_ != selected_label] = 0
            ax = imcomposite(
                a if b is None else xr.concat((a, b), dim="i"),
                dim=dim,
                figsize=figsize,
                labels_contour=labels_,
            )
            y_pos = y.start * image_spacing[0], y.stop * image_spacing[0]
            x_pos = x.start * image_spacing[0], x.stop * image_spacing[1]
            a = a.values
            b = b.values if b is not None else None

            title = [
                f"{y_pos[0]:,.1f}-{y_pos[1]:,.1f}, {x_pos[0]:,.1f}-{x_pos[1]:,.1f}"
            ]
            if y_indices is not None:
                zncc = min_zncc(a, b)
                title.append(f"ZNCC: {zncc:.4f}")
            else:
                score = sel[labels_df.columns[4]].values[0]
                title.append(f"{labels_df.columns[4]}: {score}")

            ax.set_title(", ".join(title))
            plt.show()

    if (
        y_indices is not None
        and len(y_indices) > 0
        or labels_df is not None
        and len(labels_df) > 0
    ):
        _on_index_changed(None)
    selector.observe(_on_index_changed, "value")
    return widgets.VBox([selector, chunk_out])
