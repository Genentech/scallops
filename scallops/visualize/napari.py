"""Scallop module to interact with the napari interactive viewers.

Provides functions for interacting with the Napari interactive
viewer to visualize various types of image data, including pooled in-situ
sequencing results and experiments.

Authors:
    - The SCALLOPS development team


Note:
Make sure to have Napari installed before using these functions.
"""

import os
import re
from collections import defaultdict

import dask.array as da
import dask.dataframe as dd
import fsspec
import numpy as np
import pandas as pd
import zarr
from skimage.util import img_as_float
from xarray import DataArray

from scallops.experiment.elements import Experiment
from scallops.io import read_image
from scallops.stitch._radial import radial_correct
from scallops.utils import forceTCZYX


def add_bases(
    viewer: "napari.Viewer",  # noqa: F821
    df_reads: pd.DataFrame | dd.DataFrame,
    y_slice: slice | None = None,
    x_slice: slice | None = None,
    base_colors: dict[str, str] = {
        "G": "yellow",
        "T": "magenta",
        "A": "red",
        "C": "green",
    },
    **params,
) -> "napari.layers.Points":  # noqa: F821
    """Add bases to napari viewer.

    :param viewer: The napari viewer
    :param df_reads: Dataframe containing reads
    :param y_slice: Slice over y axis
    :param x_slice: Slice over x axis
    :param base_colors: Dictionary of colors to use for bases
    :param params: Dictionary of parameters to pass to napari viewer add_points
    :return: The newly created napari points layer
    """
    y_col = "y"
    x_col = "x"
    query = []
    if y_slice is not None:
        query.append(f"`{y_col}`>={y_slice.start} & `{y_col}` < {y_slice.stop}")
    if x_slice is not None:
        query.append(f"`{x_col}`>={x_slice.start} & `{x_col}` < {x_slice.stop}")
    if len(query) > 0:
        df_reads = df_reads.query("&".join(query))
    if isinstance(df_reads, dd.DataFrame):
        df_reads = df_reads.compute()
    points = df_reads[[x_col, y_col]].values
    if y_slice is not None:
        points[:, 0] = points[:, 0] - y_slice.start
    if x_slice is not None:
        points[:, 1] = points[:, 1] - x_slice.start
    default_params = dict(
        visible=False, opacity=0.4, symbol="ring", size=10, name="bases"
    )
    params.update(default_params)
    pts_layer = viewer.add_points(
        points,
        text="base",
        properties=dict(base=df_reads["barcode"].str[0].values),
        face_color="base",
        face_color_cycle=base_colors,
        **params,
    )

    def set_pts_features(step):
        pts_layer.features["base"] = df_reads["barcode"].str[step[0]].values
        pts_layer.text = "base"
        pts_layer.face_color = "base"  # force refresh

    viewer.dims.events.current_step.connect(lambda event: set_pts_features(event.value))
    return pts_layer


def pooled_iss(
    url: str,
    viewer: "napari.Viewer" = None,  # noqa: F821
) -> "napari.Viewer":  # noqa: F821
    """View pooled in-situ sequencing results in Napari.

    Napari is a powerful multi-dimensional image viewer that allows interactive exploration
    of large and complex image datasets. This function enables you to visualize pooled in-situ
    sequencing results conveniently using Napari.

    :param url: URL to pooled in-situ sequencing results "images.zarr"
    :param viewer: An existing Napari viewer instance or None
    :return: Napari viewer

    :example:

    .. code-block:: python

        from scallops.visualize.napari import pooled_iss
        import scallops

        # URL to pooled in-situ sequencing results
        url = "path/to/images.zarr"

        # View pooled in-situ sequencing results in Napari
        napari_viewer = pooled_iss(url)

        # Show the Napari viewer
        napari_viewer.show()
    """
    import napari
    from qtpy.QtWidgets import QListWidget

    if viewer is None:
        viewer = napari.Viewer()

    fs, _ = fsspec.core.url_to_fs(url)
    url = url.rstrip(fs.sep)

    def _open_name(item=None):
        if not item:
            item = list_widget.currentItem()
        name = item.text()
        viewer.title = name
        viewer.layers.select_all()
        viewer.layers.remove_selected()
        image_keys = image_key_to_images.get(name)
        image_keys.sort()

        for i in range(len(image_keys)):
            node_name = image_keys[i]
            grp = zarr.open(url + fs.sep + "images" + fs.sep + node_name, "r")

            channel_axis = None
            ch_types = [axis["type"] for axis in grp.attrs["multiscales"][0]["axes"]]
            image_name = node_name

            if node_name != name and image_name.startswith(
                name
            ):  # keep only the suffix
                image_name = image_name[len(name) + 1 :]
            if "channel" in ch_types:
                channel_axis = ch_types.index("channel")
            if "omero" in grp.attrs:
                channel_names = [
                    channel["label"] for channel in grp.attrs["omero"]["channels"]
                ]
                image_name = [
                    image_name + "-" + channel_name for channel_name in channel_names
                ]
            params = dict(
                channel_axis=channel_axis, metadata=None, name=image_name, visible=False
            )
            if node_name == name:
                visible = []
                found = False
                for _image_name in image_name:
                    if (
                        _image_name.find("_A") != -1
                        or _image_name.find("_C") != -1
                        or _image_name.find("_T") != -1
                        or _image_name.find("_G") != -1
                    ):
                        visible.append(True)
                        found = True
                    else:
                        visible.append(False)
                params["visible"] = visible if found else True
            if node_name.endswith("-std") or node_name.endswith("-peaks"):
                params["contrast_limits"] = (0, 200)
            viewer.add_image(
                da.array(grp["0"]),
                **params,
            )

        show_contour = ["cell", "nuclei", "cytosol"]
        visible = ["cell", "spots"]
        label_keys = image_key_to_labels.get(name)

        for label_key in label_keys:
            label_suffix = label_key.split("-")[-1]
            grp = zarr.open(url + fs.sep + "labels" + fs.sep + label_key, "r")
            label_data = grp["0"]
            params = dict(name=label_key, opacity=0.5, visible=label_suffix in visible)
            if label_suffix == "spots":  # view spots as points
                label_data = label_data[...]
                params["size"] = 6
                params["edge_color"] = "white"
                params["face_color"] = np.array([1, 1, 1, 0])
                viewer.add_points(np.array(np.where(label_data > 0)).T, **params)

            else:
                label_data = da.array(label_data)
                labels = viewer.add_labels(label_data, **params)
                if label_suffix in show_contour:
                    labels.contour = 1

    list_widget = QListWidget()

    # using zarr.keys() to list keys is very slow

    # list images
    image_key_to_images = defaultdict(lambda: [])
    # e.g. 'A1-102': ['A1-102', 'A1-102-phenotype']
    image_suffixes = ["-cell-mask", "-log", "-max", "-peaks", "-std", "-phenotype"]
    for key in fs.ls(url + fs.sep + "images"):
        key = os.path.basename(key)
        if key[0] != ".":
            details = None
            for suffix in image_suffixes:
                if key.endswith(suffix):
                    details = suffix
                    break
            root_key = key if details is None else key[: -len(details)]
            image_key_to_images[root_key].append(key)

    # list labels
    image_key_to_labels = defaultdict(lambda: [])
    # e.g. 'A1-102': ['A1-102-spots', 'A1-102-cell', 'A1-102-nuclei', 'A1-102-cytosol']
    label_suffixes = ["-cell", "-cytosol", "-nuclei", "-spots"]
    for key in fs.ls(url + fs.sep + "labels"):
        key = os.path.basename(key)
        if key[0] != ".":
            for suffix in label_suffixes:
                if key.endswith(suffix):
                    details = suffix
                    break
            root_key = key if details is None else key[: -len(details)]
            image_key_to_labels[root_key].append(key)

    for key in image_key_to_images.keys():
        list_widget.addItem(key)
    list_widget.currentItemChanged.connect(_open_name)
    viewer.window.add_dock_widget([list_widget], area="right", name="image")
    return viewer


def experiment_napari(
    experiment: Experiment,
    labels: Experiment | dict[str, np.ndarray | DataArray] | None = None,
    title_attribute: str = "common_src",
    viewer: "napari.Viewer" = None,  # noqa: F821
    **kwargs,
) -> "napari.Viewer":  # noqa: F821
    """View an exp in Napari.

    Napari is a versatile multi-dimensional image viewer, and this function facilitates
    the visualization of an entire exp, including different imaging channels and timepoints,
    using the Napari viewer.

    :param experiment:
        Experiment to display
    :param labels:
        Labels (e.g., from segmentation) that map image key to label array.
    :param title_attribute:
        Attribute to set as the window title
    :param viewer:
        An existing Napari viewer instance
    :param kwargs:
        Keyword arguments to be passed to viewer.add_image
    :return:
        Napari viewer

    :example:

    .. code-block:: python

        from scallops.visualize.napari import expnapari
        import scallops

        # Load an exp
        exp = scallops.io.read_experiment("path/to/exp")

        # View the exp in Napari
        napari_viewer = expnapari(exp)
    """
    import napari
    from qtpy.QtWidgets import QListWidget

    if viewer is None:
        viewer = napari.Viewer()

    def _open_name(item):
        name = item.text()
        viewer.title = name
        viewer.layers.select_all()
        viewer.layers.remove_selected()
        label = None
        if labels is not None:
            if isinstance(labels, Experiment):
                label = labels.labels[name]
            else:
                label = labels.get(name)
        imnapari(
            image=experiment.images[name],
            viewer=viewer,
            title_attribute=title_attribute,
            labels=label,
            **kwargs,
        )

    list_widget = QListWidget()
    for n in experiment.images.keys():
        list_widget.addItem(n)

    list_widget.currentItemChanged.connect(_open_name)

    viewer.window.add_dock_widget([list_widget], area="right")
    list_widget.setCurrentRow(0)
    return viewer


def imnapari(
    image: DataArray | dict[str, DataArray] | None,
    labels: None | np.ndarray | DataArray | dict[str, np.ndarray | DataArray] = None,
    points: pd.DataFrame | None = None,
    title_attribute: str = "common_src",
    viewer: "napari.Viewer" = None,  # noqa: F821
    point_size: int = 5,
    **kwargs,
) -> "napari.Viewer":  # noqa: F821
    """View image in Napari.

    Napari is a versatile multi-dimensional image viewer, and this function facilitates
    the visualization of a single image along with optional labels in the Napari viewer.

    :param point_size: Size of points if provided
    :param points: Dataframe with peaks information
    :param image: Image(s) to display. If a dictionary, the key would be the display name
    :param labels: Labels (e.g., from segmentation) that map label name to label array.
    :param title_attribute: Attribute to set as the window title
    :param viewer: An existing Napari viewer instance
    :param kwargs: Keyword arguments to be passed to viewer.add_image
    :return: Napari viewer

    :example:

        .. code-block:: python

            import xarray as xr
            import numpy as np
            from napari import Viewer
            from scallops.visualize.napari import imnapari

            # Create a synthetic image
            width = 512
            height = 512
            channels = 3
            data = np.random.randint(
                0, 255, size=(channels, height, width), dtype=np.uint8
            )
            coords = {
                "c": np.arange(channels),
                "y": np.arange(height),
                "x": np.arange(width),
            }
            synthetic_image = xr.DataArray(data, coords=coords, dims=("c", "y", "x"))

            # Create a Napari viewer
            viewer = Viewer()

            # View the synthetic image in Napari
            imnapari(synthetic_image, title_attribute="Synthetic Image", viewer=viewer)

            # Run the Napari event loop
            viewer.show()
    """
    import napari

    images = None

    if isinstance(image, DataArray):
        images = {"Image": forceTCZYX(image)}
    elif isinstance(image, dict):
        images = {k: forceTCZYX(v) for k, v in image.items()}

    if viewer is None:
        viewer = napari.Viewer()
    if image is not None:
        for name, image in images.items():
            axis_labels = []
            channel_axis = 1
            for i, x in enumerate(image.dims):
                if x == "c":
                    channel_axis = i
                else:
                    axis_labels.append(x)

            channel_labels = [f"{name}::Channel {x}" for x in image.coords["c"].values]

            viewer.add_image(
                image,
                channel_axis=channel_axis,
                name=channel_labels,
                metadata=image.attrs,
                **kwargs,
            )
            viewer.dims.axis_labels = axis_labels
            viewer.title = image.attrs.get(title_attribute, "")

    if labels is not None:
        if isinstance(labels, (np.ndarray, DataArray)):
            viewer.add_labels(labels, name="labels", opacity=0.5)
        else:
            for label_name, label_image in labels.items():
                viewer.add_labels(label_image, name=label_name, opacity=0.5)
    if points is not None:
        from magicgui import magicgui

        assert any(points.columns.isin(["x", "y"])), (
            "Dataframe must contain at least x and y coordinates"
        )

        feats = points.drop(columns=["x", "y"]).to_dict(orient="list")

        viewer.add_points(
            points.loc[:, ["y", "x"]],
            face_color="transparent",
            edge_color="yellow",
            features=feats,
            properties=points.drop(columns=["x", "y"]),
            size=point_size,
        )

        for feat, values in feats.items():
            mini, maxi = min(values), max(values)
            dtype = re.sub("[0-9]", "", points[feat].dtype.name)
            dtype = str if dtype == "object" else exec(dtype)

            @magicgui(
                call_button=f"{feat} filter",
                slider={"widget_type": "FloatSlider", "min": mini, "max": maxi},
                auto_call=True,
            )
            def slider(slider: dtype) -> None:
                filtered_indices = [
                    i for i, val in enumerate(feats[feat]) if val > slider
                ]
                viewer.layers[-1].data = points.iloc[filtered_indices][
                    ["y", "x"]
                ].values

            viewer.window.add_dock_widget(slider)

    return viewer


def radial_distortion_estimation(
    top_left: str,
    top_right: str,
    bottom_left: str,
    bottom_right: str,
    channel: int = 0,
    proportion_overlap: float = 0.0,
) -> None:
    """Run radial distortion estimation. This function initializes a Napari viewer with four images
    positioned to form a larger composite image. It applies radial distortion correction to the
    images, allowing the user to adjust the correction parameter interactively through a graphical
    user interface. Images must be adjacent.

    :param top_left: Path to the top-left image file.
    :param top_right: Path to the top-right image file.
    :param bottom_left: Path to the bottom-left image file.
    :param bottom_right: Path to the bottom-right image file.
    :param channel: Index of the channel to be extracted from the image files.
    :param proportion_overlap: Proportion of overlap between adjacent images, expressed as a fraction
                               of the image size. This is used to compute the translations for positioning
                               the images within the viewer.

    :example:
        To run the distortion correction process, use the following code with your own S3 URIs:

        .. code-block:: python

            radial_distortion_estimation(
                top_left="s3://your-bucket/path/to/image_top_left.nd2",
                top_right="s3://your-bucket/path/to/image_top_right.nd2",
                bottom_left="s3://your-bucket/path/to/image_bottom_left.nd2",
                bottom_right="s3://your-bucket/path/to/image_bottom_right.nd2",
                channel=0,
                proportion_overlap=0.05,
            )

        This should open a napari window with the 4 images displayed. Adjust the alignment of the images, and use the
        `Apply Distortion Correction` slider to identify the K for radial correction. You can pass that to the stitch
        command of scallops or ashlar.
    """
    import napari
    from magicgui import magicgui
    from qtpy.QtGui import QGuiApplication
    from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

    def _read_as_foat(img, channel):
        img = read_image(img).isel(c=channel).squeeze()
        img.data = img_as_float(img.data)
        return img

    command_text = ""

    images = {
        "top_left": _read_as_foat(top_left, channel),
        "top_right": _read_as_foat(top_right, channel),
        "bottom_left": _read_as_foat(bottom_left, channel),
        "bottom_right": _read_as_foat(bottom_right, channel),
    }

    @magicgui(
        call_button="Apply Distortion Correction",
        log10k={"widget_type": "FloatSlider", "min": -10, "max": -6, "step": 0.01},
    )
    def apply_distortion(log10k: float = -8) -> str:
        """Apply radial distortion correction to images in the viewer."""
        nonlocal command_text  # Allow modification of the outer variable
        K = 10**log10k
        for layer in viewer.layers:
            corrected_image = radial_correct(images[layer.name].data, K)
            layer.data = corrected_image
        command_text = f"--radial-correction-k={K:.4g}"
        correction_label.setText(
            f"If you are satisfied with the correction, add to your stitch command line:"
            f"<br><br><code>{command_text}</code>"
        )
        print(f"Estimated K = {K}")
        return command_text

    def _copy_to_clipboard(command: str) -> None:
        clipboard = QGuiApplication.clipboard()
        clipboard.setText(command)
        print("Command copied to clipboard")

    images_meta = {k: v.attrs["ome"]["images"][0]["pixels"] for k, v in images.items()}
    translations = [
        (0, 0),
        (
            0,
            images_meta["top_left"]["size_x"]
            - (images_meta["top_right"]["size_x"] * proportion_overlap),
        ),
        (
            images_meta["top_left"]["size_y"]
            - (images_meta["bottom_left"]["size_y"] * proportion_overlap),
            0,
        ),
        (
            images_meta["top_left"]["size_y"]
            - (images_meta["bottom_left"]["size_y"] * proportion_overlap),
            images_meta["top_left"]["size_x"]
            - (images_meta["top_right"]["size_x"] * proportion_overlap),
        ),
    ]
    viewer = napari.Viewer()
    common = dict(blending="additive", interpolation2d="bicubic")
    for i, (name, color) in enumerate(
        zip(
            ["top_left", "top_right", "bottom_left", "bottom_right"],
            ["red", "cyan", "magenta", "yellow"],
        )
    ):
        viewer.add_image(
            images[name].data,
            name=name,
            translate=translations[i],
            colormap=color,
            **common,
        )

    widget = QWidget()
    layout = QVBoxLayout()
    correction_label = QLabel("")
    correction_label.setWordWrap(True)
    layout.addWidget(apply_distortion.native)
    layout.addWidget(correction_label)
    copy_button = QPushButton("Copy to Clipboard")
    copy_button.clicked.connect(lambda: _copy_to_clipboard(command_text))
    layout.addWidget(copy_button)
    widget.setLayout(layout)
    viewer.window.add_dock_widget(widget, area="right")

    napari.run()
    return viewer
