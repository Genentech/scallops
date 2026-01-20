"""Scallop's module for Basic illumination correction.

This module provides functions for running Basic illumination correction on image data.
Basic illumination correction is a method for correcting uneven illumination in microscopy images.

Note:
Before using this module, make sure to have the required dependencies installed, including `basicpy` and `jax`.

Authors:
    - The SCALLOPS development team.
"""

import logging
from collections.abc import Mapping
from typing import Literal

import fsspec
import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from skimage.transform import resize as skimage_resize
from xarray import DataArray

from scallops.io import _get_fs_protocol, read_image, save_ome_tiff
from scallops.xr import _z_projection

logger = logging.getLogger("scallops")
try:
    import jax.numpy as jnp
    from basicpy import BaSiC
    from jax import device_put
    from jax.image import ResizeMethod
    from jax.image import resize as jax_resize
except ModuleNotFoundError:
    logger.info("Please install basicpy")


def _load_image(
    path: str, z: int | str, working_size: int | None
) -> tuple[np.ndarray, tuple[int, int]]:
    """Load and preprocess an image from the given path.

    This function reads an image, selects the appropriate z-plane or computes the
    maximum projection along the z-axis, and optionally resizes the image to a specified working size.

    :param path: Path to the image file.
    :param z: Z-plane index or `"max"` for maximum projection along the z-axis.
    :param working_size: If provided, resize the image to this size (preserving the aspect ratio).
    :return: A tuple containing the processed image array and its original shape.

    :raises AssertionError: If the image contains more than one time point.

    :example:

    .. code-block:: python

        image, original_shape = _load_image(
            "path/to/image.ome.tif", z="max", working_size=512
        )
        print(image.shape)  # Output: (1, 512, 512)
        print(original_shape)  # Output: (1024, 1024)
    """
    image = read_image(path)

    if "t" in image.sizes:
        assert image.sizes["t"] == 1, "Expected a single time point in the image."
    image = _z_projection(image, z).squeeze(("t", "z"))
    image_shape = image.shape[-2:]
    if working_size is not None:
        resize_params = dict(method=ResizeMethod.LINEAR)
        working_shape = [working_size] * (image.ndim - 1)
        target_shape = [*image.shape[:1], *working_shape]
        image = jax_resize(
            device_put(image.values).astype(jnp.float32), target_shape, **resize_params
        )
    return image, image_shape


def basic_illumination_correction(
    pattern: str | list[str],
    max_images: int | None = None,
    n_jobs: int = -1,
    output_path: str = ".",
    plot_fit: bool = True,
    z: Literal["max"] | int = "max",
    working_size: int = 128,
    **basic_kwargs: Mapping,
) -> None:
    """Run illumination correction with BasiCpy.

    :param pattern: File pattern(s) of images to include.
    :param max_images: Use only this maximum number of images as input to BaSiC.
    :param n_jobs: Number of jobs to run in parallel for processing channels.
    :param output_path: Output path to save results.
    :param plot_fit: Whether to plot fit.
    :param z: Either 'max' or a z-index.
    :param working_size: Resize images to specified size.
    :param basic_kwargs: Arguments to pass to BaSiC.

    :return: None.

    :raises AssertionError: If no images are found.

    :note: This function requires the 'basicpy', 'jax', 'device_put', and 'jax.image' libraries.
    Install them using `pip install basicpy jax jaxlib`.

    :example:

    .. code-block:: python
        from scallops.basic import basic_illumination_correction

        # Specify the file pattern for the input images
        pattern = "path/to/images/*.tif"

        # Run Basic illumination correction
        basic_illumination_correction(
            pattern,
            max_images=10,
            n_jobs=-1,
            output_path="output",
            plot_fit=True,
            z="max",
            working_size=128,
            additional_basic_kwargs={"get_darkfield": True},
        )
    """
    if isinstance(pattern, str):
        pattern = [pattern]
    files = []
    for p in pattern:
        fs, _ = fsspec.core.url_to_fs(p)
        matches = fs.glob(p)
        if _get_fs_protocol(fs) != "file":
            matches = [f"{_get_fs_protocol(fs)}://{x}" for x in matches]
        files += matches
    if max_images is not None and len(files) > max_images:
        rng = np.random.default_rng()
        indices = rng.choice(len(files), max_images, replace=False)
        files = [files[index] for index in indices]
    #  (T,Y,X) or (T,Z,Y,X). T can be either of time or mosaic position.
    assert len(files) > 0, "No images found"
    fs, _ = fsspec.core.url_to_fs(output_path)
    root = output_path.rstrip(fs.sep)
    if not fs.exists(root):
        fs.mkdir(root)

    image_tuples = Parallel(n_jobs=-1)(
        delayed(_load_image)(f, z, working_size) for f in files
    )
    images = [image_tuple[0] for image_tuple in image_tuples]
    full_image_shape = image_tuples[0][1]
    image = np.array(images)
    nchannels = image.shape[1]
    _basic_kwargs = dict(get_darkfield=True)
    _basic_kwargs.update(basic_kwargs)
    models = Parallel(n_jobs=n_jobs)(
        delayed(_basic_single_channel)(image[:, i], **_basic_kwargs)
        for i in range(nchannels)
    )
    flatfields = []
    darkfields = []
    model_dir = root + fs.sep + "model"
    if not fs.exists(model_dir):
        fs.mkdir(model_dir)
    if plot_fit:
        pdf = PdfPages(f"{root}{fs.sep}illumination-correction.pdf")
    for i in range(len(models)):
        model = models[i]
        model_path = model_dir + fs.sep + f"c{i}"
        if fs.exists(model_path):
            fs.rm(model_path, recursive=True)
        model.save_model(model_path)
        flatfields.append(model.flatfield)
        if model.darkfield is not None:
            darkfields.append(model.darkfield)

        if plot_fit:
            basic_qc_plot(model.flatfield, model.darkfield, model.baseline)
            plt.suptitle(f"Channel {i}")
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()

    if plot_fit:
        pdf.close()

    flatfield = np.array(flatfields)
    if working_size is not None:
        flatfield = skimage_resize(
            flatfield, (nchannels, full_image_shape[0], full_image_shape[1])
        )

    save_ome_tiff(
        data=flatfield, dim_order="CYX", uri=f"{root}{fs.sep}flatfield.ome.tiff"
    )
    if len(darkfields) > 0:
        darkfield = np.array(darkfields)
        if working_size is not None:
            darkfield = skimage_resize(
                darkfield, (nchannels, full_image_shape[0], full_image_shape[1])
            )
        save_ome_tiff(
            data=darkfield, dim_order="CYX", uri=f"{root}{fs.sep}darkfield.ome.tiff"
        )


def _basic_single_channel(
    image: np.ndarray | DataArray, **basic_kwargs: dict
) -> object:
    """Apply the BaSiC algorithm to fit illumination correction on a single-channel image.

    This function initializes the BaSiC algorithm with the provided keyword arguments,
    fits the model to the given image, and returns the fitted BaSiC object.

    :param image: A 2D or 3D array representing the single-channel image.
                  Can be a NumPy array or an xarray DataArray.
    :param basic_kwargs: Additional keyword arguments to configure the BaSiC algorithm.
    :return: The fitted BaSiC object.

    :example:

    .. code-block:: python

        from basicpy import BaSiC  # Example import of the BaSiC algorithm

        image = np.random.rand(512, 512)  # Example image
        basic_obj = _basic_single_channel(image, max_iterations=100)
        print(basic_obj)  # Output: <BaSiC object with fitted parameters>
    """
    # Initialize the BaSiC algorithm with provided arguments
    basic = BaSiC(**basic_kwargs)

    # Fit the model to the input image
    basic.fit(image)

    # Return the fitted BaSiC object
    return basic


def basic_qc_plot(
    flatfield: np.ndarray,
    darkfield: np.ndarray | None = None,
    baseline: np.ndarray | None = None,
) -> None:
    """Generate a quality control (QC) plot for BaSiC illumination correction.

    :param flatfield: Flatfield array obtained from BaSiC correction.
    :param darkfield: Darkfield array obtained from BaSiC correction (optional).
    :param baseline: Baseline array obtained from BaSiC correction (optional).

    :return: None.

    :example:

    .. code-block:: python
        from scallops.basic import basic_qc_plot
        import numpy as np

        # Example flatfield, darkfield, and baseline arrays
        flatfield_array = np.random.rand(128, 128)
        darkfield_array = np.random.rand(128, 128)
        baseline_array = np.random.rand(10)

        # Generate a QC plot
        basic_qc_plot(flatfield_array, darkfield_array, baseline_array)
    """
    ncols = 3
    if darkfield is None:
        ncols -= 1
    if baseline is None:
        ncols -= 1
    fig, axes = plt.subplots(1, ncols, figsize=(9, 3))
    if isinstance(axes, plt.Axes):
        axes = [axes]
    im = axes[0].imshow(flatfield)
    fig.colorbar(im, ax=axes[0])
    axes[0].set_title("Flatfield")

    if darkfield is not None:
        im = axes[1].imshow(darkfield)
        fig.colorbar(im, ax=axes[1])
        axes[1].set_title("Darkfield")

    if baseline is not None:
        col = 2 if darkfield is not None else 1
        axes[col].plot(baseline)
        axes[col].set_xlabel("Frame")
        axes[col].set_ylabel("Baseline")
    fig.tight_layout()
