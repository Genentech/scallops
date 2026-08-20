"""The `codebook` module provides functions for decoding features from a codebook and estimating
scaling factors to equalize bit intensities.

Authors:
    - The SCALLOPS development team.
"""

from collections.abc import Hashable, Sequence
from typing import Literal

import numpy as np
import pandas as pd
import skimage.measure
import xarray as xr
from skimage.measure import label, regionprops
from sklearn.metrics.pairwise import pairwise_distances_argmin_min


def reshape_image_to_codes(array: xr.DataArray) -> np.ndarray:
    """Reshape a 5-dimensional array of shape (t, c, z, y, x) to a 2-dimensional array of shape
    (z+y+x, t+c).

    :param array: The input 5-dimensional array representing images. Dimensions: (t, c, z, y, x).
    :return: Reshaped 2-dimensional array of shape (z+y+x, t+c).
    """
    # Rearrange axes to z, y, x, t, c and then reshape to 2D array of (z+y+x, t+c)
    dims = ["y", "x", "t", "c"]
    if "z" in array.dims:
        dims = ["z"] + dims
    return array.transpose(*dims).values.reshape(
        -1, array.sizes["t"] * array.sizes["c"]
    )


def _regionprops(
    argmin: np.ndarray,
    distances: np.ndarray,
    passes_filters: np.ndarray,
    sizes: dict[Hashable, int],
) -> tuple[np.ndarray, np.ndarray, list[object]]:
    """Compute labeled regions and their properties for the given input arrays.

    This function labels the regions in the input `argmin` array, filtered by
    the `passes_filters` mask. If the input includes a "z" dimension, the arrays
    are reshaped accordingly. It returns the labeled image, reshaped distances,
    and the properties of the labeled regions.

    :param argmin: An array representing the minimum indices for each pixel.
    :param distances: An array of distances corresponding to each pixel.
    :param passes_filters: A boolean mask indicating which pixels pass the filters.
    :param sizes: A dictionary specifying the size of each dimension
                  (e.g., {"y": 512, "x": 512} or {"z": 10, "y": 512, "x": 512}).

    :return: A tuple containing:
             - The filtered and reshaped `argmin` array.
             - The reshaped `distances` array.
             - A list of region properties for the labeled regions.
    """
    # Copy argmin to prevent modifying the original array
    argmin_ = argmin.copy()
    # Assign -1 to positions that do not pass the filters
    argmin_[~passes_filters] = -1
    # Determine if the data contains a "z" dimension
    has_z = "z" in sizes
    # Reshape arrays based on the presence of the "z" dimension
    if has_z:
        argmin_ = argmin_.reshape((sizes["z"], sizes["y"], sizes["x"]))
        distances_ = distances.reshape((sizes["z"], sizes["y"], sizes["x"]))
    else:
        argmin_ = argmin_.reshape((sizes["y"], sizes["x"]))
        distances_ = distances.reshape((sizes["y"], sizes["x"]))
    # Label the regions in the filtered argmin array
    labeled_image = label(argmin_, background=-1)
    # Compute region properties without caching for efficiency
    props = regionprops(labeled_image, cache=False)

    return argmin_, distances_, props


def decode_image(
    array: xr.DataArray,
    codebook: xr.DataArray,
    metric: Literal[
        "braycurtis",
        "canberra",
        "chebyshev",
        "cityblock",
        "euclidean",
        "haversine",
        "infinity",
        "kulsinski",
        "l1",
        "l2",
        "manhattan",
        "matching",
        "minkowski",
        "p",
        "seuclidean",
        "sqeuclidean",
    ] = "euclidean",
    norm_order: int = 2,
    scale_factors: np.ndarray = None,
    min_intensity: float = 0,
    max_distance: float = np.inf,
) -> pd.DataFrame:
    """Call features using the supplied codebook.

    :param array: 5-d array with dimensions t, c, z, y, x
    :param codebook: The codebook used to call features. Dimensions are f, t, c (feature, time,
        channel)
    :param norm_order: Norm to apply (`numpy:reference/generated/numpy.linalg.norm`)
    :param metric: Distance metric
    :param min_intensity: Minimum intensity to include
    :param max_distance: Maximum distance between a feature and its closest code for which the coded
        target will be assigned.
    :param scale_factors: Optional 1-d array (time, channel) to divide array by
    :return: Data frame containing called features
    """
    codes = reshape_image_to_codes(array)
    argmin, distances, trace_norms, passes_filters = _decode_metric(
        array=codes,
        codebook=codebook,
        metric=metric,
        norm_order=norm_order,
        scale_factors=scale_factors,
        min_intensity=min_intensity,
        max_distance=max_distance,
    )
    argmin_, distances_, props = _regionprops(
        argmin, distances, passes_filters, array.sizes
    )
    return _regionprops_to_table(
        argmin_=argmin_,
        distances_=distances_,
        codebook=codebook,
        props=props,
        has_z="z" in array.dims,
    )


def _regionprops_to_table(
    argmin_: np.ndarray,
    distances_: np.ndarray,
    codebook: xr.DataArray,
    props: list[skimage.measure._regionprops.RegionProperties],
    has_z: bool = False,
) -> pd.DataFrame:
    """Convert region properties and related data into a pandas DataFrame.

    This function processes the labeled regions and extracts relevant properties
    such as coordinates, area, distances, and features, creating a DataFrame that
    summarizes these attributes.

    :param argmin_: Array representing the index of the closest codebook entry for each pixel.
    :param distances_: Array of distances for each pixel in the labeled regions.
    :param codebook: A DataArray representing the codebook used to map features.
    :param props: A list of `RegionProperties` objects containing information about labeled regions.
    :param has_z: Whether the input data includes a "z" dimension (3D) (default is False).

    :return: A DataFrame containing the following columns:
        - `y`: Y-coordinate of the centroid.
        - `x`: X-coordinate of the centroid.
        - `mean_distance`: Mean distance of the pixels in the region to the nearest codebook entry.
        - `min_distance`: Minimum distance within the region.
        - `area`: Area (number of pixels) in the region.
        - `feature`: The feature corresponding to the codebook index.
        - `z` (optional): Z-coordinate of the centroid (if `has_z` is True).

    :example:

    .. code-block:: python

        # Example usage
        argmin_ = np.random.randint(0, 5, (100, 100))
        distances_ = np.random.random((100, 100))
        codebook = xr.DataArray(np.array(["A", "B", "C", "D", "E"]), dims=["f"])
        props = regionprops(label(argmin_))

        df = _regionprops_to_table(argmin_, distances_, codebook, props, has_z=False)
        print(df.head())
    """
    # Initialize arrays for storing region properties
    areas = np.zeros(len(props), dtype=int)
    x = np.zeros(len(props))
    y = np.zeros(len(props))

    if has_z:
        z = np.zeros(len(props))

    mean_distance = np.zeros(len(props))
    min_distance = np.zeros(len(props))
    features = []

    # Iterate through each region property
    for i, p in enumerate(props):
        coords = p.coords  # Coordinates of the region's pixels
        areas[i] = p.area  # Region area
        centroid = p.centroid  # Region centroid

        if has_z:
            codebook_index = argmin_[coords[0][0], coords[0][1], coords[0][2]]
            p_distances = distances_[coords[:, 0], coords[:, 1], coords[:, 2]]
            z[i], y[i], x[i] = centroid  # Store Z, Y, X coordinates
        else:
            codebook_index = argmin_[coords[0][0], coords[0][1]]
            p_distances = distances_[coords[:, 0], coords[:, 1]]
            y[i], x[i] = centroid  # Store Y, X coordinates

        mean_distance[i] = p_distances.mean()  # Mean distance of the region's pixels
        min_distance[i] = p_distances.min()  # Minimum distance in the region

        # Append the corresponding feature from the codebook
        features.append(codebook.f.values[codebook_index])

    # Create a DataFrame from the collected data
    df = pd.DataFrame(
        data=dict(
            y=y,
            x=x,
            mean_distance=mean_distance,
            min_distance=min_distance,
            area=areas,
            feature=features,
        )
    )

    # Include the Z-coordinate if applicable
    if has_z:
        df["z"] = z

    return df


def _decode_metric(
    array: np.ndarray,
    codebook: xr.DataArray,
    metric: str = "euclidean",
    norm_order: int = 2,
    scale_factors: np.ndarray | None = None,
    min_intensity: float = 0,
    max_distance: float = np.inf,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Call features using the supplied codebook.

    :param array: 2-d array. First dimension is read, the second time and channel
    :param codebook: The codebook used to call features. Dimensions are f, t, c (feature, time,
        channel)
    :param norm_order: Norm to apply (`numpy:reference/generated/numpy.linalg.norm`)
    :param metric: Distance metric
    :param min_intensity: Minimum normed value in array to include
    :param max_distance: Maximum distance between a feature and its closest code for which the coded
        target will be assigned.
    :param scale_factors: Optional 1-d array (time, channel) to divide array by
    :return: Tuple of called indices, distances to the closest feature, value norms, and whether
        called feature passes min intensity and max distance filters
    """

    codebook_pixel_array = codebook.stack(pixel_array=("t", "c"))
    codebook_pixel_array, _ = unit_norm(codebook_pixel_array, norm_order=norm_order)
    if scale_factors is None:
        scale_factors = np.ones(codebook_pixel_array.shape[1])
    normed_trace, norms = unit_norm(array / scale_factors, norm_order=norm_order)

    argmin, distances = pairwise_distances_argmin_min(
        normed_trace, codebook_pixel_array, metric=metric
    )

    # keep features with low distance and high intensity
    passes_filters = np.logical_and(
        norms >= min_intensity, distances <= max_distance, dtype=bool
    )
    return argmin, distances, norms, passes_filters


def unit_norm(
    array: np.ndarray | xr.DataArray,
    norm_order: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Unit normalize each feature.

    :param array: 2-dimensional array. First dimension is f (feature), the second t+c.
    :param norm_order: Norm to apply (`numpy:reference/generated/numpy.linalg.norm`).
    :return: Normalized array and norms.

    :example:

    .. code-block:: python

        import numpy as np
        from scallops.codebook import unit_norm

        # Create a synthetic array
        array = np.random.rand(5, 10)

        # Apply unit normalization
        normalized_array, norms = unit_norm(array)
    """
    _array = array.values.copy() if isinstance(array, xr.DataArray) else array.copy()
    _array[np.abs(array).sum(axis=1) == 0] = 1
    norm = np.linalg.norm(_array, ord=norm_order, axis=1)
    return _array / norm[:, None], norm


def estimate_scale_factors(
    codes: xr.DataArray | np.ndarray,
    codebook: xr.DataArray,
    initial_scale_factors: np.ndarray = None,
    max_iter: int = 10,
    metric: Literal[
        "braycurtis",
        "canberra",
        "chebyshev",
        "cityblock",
        "euclidean",
        "haversine",
        "infinity",
        "kulsinski",
        "l1",
        "l2",
        "manhattan",
        "matching",
        "minkowski",
        "p",
        "seuclidean",
        "sqeuclidean",
    ] = "euclidean",
    norm_order: int = 2,
    min_intensity: float = 0,
    max_distance: float = np.inf,
) -> np.ndarray:
    """Calculate the scale factors that would result in the mean on bit intensity for each bit to be
    equal.

    :param codes: 2-dimensional array with dimensions read and t+c.
    :param codebook: The codebook used to call features. Dimensions are f (feature), t, c.
    :param norm_order: Norm to apply (`numpy:reference/generated/numpy.linalg.norm`).
    :param metric: Distance metric as in `sklearn.metrics.pairwise.ArgKmin.valid_metrics()`.
    :param min_intensity: Minimum intensity to include.
    :param max_distance: Maximum distance between a feature and its closest code for which the coded target will be
        assigned.
    :param initial_scale_factors: Initial scale factors of 1-dimensional array (t+c) to divide array by. If not provided, set to 90th percentile
        for each bit.
    :param max_iter: Maximum number of iterations to perform.
    :return: The estimated scaling factors.
    """
    if not isinstance(codes, xr.DataArray):
        codes = xr.DataArray(codes)
    scale_factors = initial_scale_factors
    if scale_factors is None:
        scale_factors = np.nanquantile(codes, 0.9, axis=0)
        if np.any(scale_factors == 0):
            scale_factors = np.nanmax(codes, axis=0)
    codebook_pixel_array = codebook.stack(pixel_array=("t", "c"))  # dims (f,t+c)
    codebook_pixel_array, _ = unit_norm(codebook_pixel_array, norm_order=norm_order)
    codebook_pixel_array = codebook_pixel_array > 0

    for i in range(max_iter):
        argmin, distances, values_norms, passes_filters = _decode_metric(
            array=codes,
            codebook=codebook,
            scale_factors=scale_factors,
            metric=metric,
            norm_order=norm_order,
            min_intensity=min_intensity,
            max_distance=max_distance,
        )

        new_scale_factors = _update_scale_factors(
            codes, passes_filters, values_norms, codebook_pixel_array, argmin
        )
        # new_scale_factors /= new_scale_factors.min()
        if np.all(new_scale_factors == 1) or np.array_equal(
            new_scale_factors, scale_factors
        ):
            break
        scale_factors = new_scale_factors

    return scale_factors


def _update_scale_factors(
    array: np.ndarray | xr.DataArray,
    passes_filters: np.ndarray,
    values_norms: np.ndarray,
    codebook_pixel_array: np.ndarray | xr.DataArray,
    argmin: np.ndarray,
) -> np.ndarray:
    """Update scale factors based on normalized values and bit indicators.

    This function calculates new scale factors by normalizing the input array's
    values and computing the mean for each bit position. If NaNs occur, they are
    replaced with a value of 1.

    :param array: Input array containing the data to be scaled.
    :param passes_filters: A boolean mask indicating which elements pass the filter criteria.
    :param values_norms: Array of normalization values for scaling the input array.
    :param codebook_pixel_array: Codebook pixel array indicating bit positions.
    :param argmin: Indices indicating the minimum values for each row in the array.

    :return: A 1D array of updated scale factors.

    :example:

    .. code-block:: python

        array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        passes_filters = np.array([True, False, True])
        values_norms = np.array([1, 1, 2])
        codebook_pixel_array = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        argmin = np.array([0, 1, 2])

        new_factors = _update_scale_factors(
            array, passes_filters, values_norms, codebook_pixel_array, argmin
        )
        print(new_factors)  # Output: [1. 1. 1.]

    :raises ValueError: If the shapes of the inputs are incompatible.
    """
    scaled_values = array[passes_filters] / np.expand_dims(
        values_norms[passes_filters], 1
    )
    # create mask for each row in array, indicating where on bits are located
    if isinstance(codebook_pixel_array, xr.DataArray):
        on_bit_indicator = codebook_pixel_array[argmin].values[passes_filters]
    else:
        on_bit_indicator = codebook_pixel_array[argmin][
            passes_filters
        ]  # same shape as array
    on_mean_per_bit = scaled_values.where(on_bit_indicator).mean(axis=0).values

    new_scale_factors = on_mean_per_bit / np.nanmean(on_mean_per_bit)
    new_scale_factors[np.isnan(new_scale_factors)] = 1
    return new_scale_factors


def barcodes_to_codebook(
    barcodes: Sequence[str],
    features: Sequence[str],
    encoding: np.ndarray,
    encoding_bases: Sequence[str],
):
    # encoding = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
    encoding_base_map = dict()
    for i in range(len(encoding_bases)):
        encoding_base_map[encoding_bases[i]] = i
    n_cycles = len(barcodes[0])
    codebook = np.zeros((len(features), n_cycles, 3), dtype=np.uint32)

    for barcode_index in range(len(barcodes)):
        barcode = barcodes[barcode_index]
        for cycle_index in range(n_cycles):
            base_index = encoding_base_map[barcode[cycle_index]]
            encoded_value = encoding[base_index]
            codebook[barcode_index, cycle_index] = encoded_value

    coords = dict(f=features)
    return xr.DataArray(codebook, dims=("f", "t", "c"), coords=coords)
