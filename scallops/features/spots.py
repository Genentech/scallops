import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from filelock import FileLock

from scallops.segmentation.util import _download_model

logger = logging.getLogger("scallops")

local_model_dest = Path.home() / ".ufish"
weights_path = local_model_dest / "v1.0-alldata-ufish_c32.pth"


def _download_ufish_model():
    with FileLock(".scallops-ufish.lock"):
        model_dir = os.environ.get("SCALLOPS_MODEL_DIR")
        if not weights_path.exists():
            if model_dir is not None and model_dir != "":
                _download_model(local_model_dest, weights_path.name)
            else:
                from ufish.api import UFish
                from ufish.utils.log import logger as ufish_logger

                model = UFish(default_weights_file="v1.0-alldata-ufish_c32.pth")
                ufish_logger.remove()
                model.load_weights(weights_file="v1.0-alldata-ufish_c32.pth")
    os.remove(".scallops-ufish.lock")


def _load_ufish_model():
    from ufish.api import UFish
    from ufish.utils.log import logger as ufish_logger

    model = UFish()
    ufish_logger.remove()
    model.load_weights_from_path(weights_path)
    return model


def spot_count(
    c: Sequence[int],
    intensity_threshold=0.06,
    channel_names: Sequence[str] = None,
    unique_labels: np.ndarray = None,
    label_image: np.ndarray = None,
    intensity_image: np.ndarray = None,
    model: Optional["ufish.api.UFish"] = None,  # noqa
    connectivity=2,
    laplace_process=True,
    axes="yx",
    return_spots_coords=False,
    **kwargs,
) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Determine the number of FISH spots within labeled regions for specified channels.

    This function identifies FISH spots in microscopy images by enhancing the image using
    the UFish model and then applying a local peak detection algorithm. It then counts
    how many of these detected spots fall within non-zero regions of a provided label mask.
    The counts are returned per channel as a dictionary, where each value is a 1D NumPy
    array. The array's length is equal to the number of unique labels, and its index `i`
    corresponds to the count for the cell (label) with ID `unique_labels[i]`. Specifically,
    if a spot falls into a region with label `L`, its count contributes to `counts[L - 1]`.

    :param c: Sequence of 0-based channel indices for which to count spots.
    :param channel_names: Sequence of strings, where each string is the name of a channel,
        corresponding to the channel indices in `intensity_image`.
    :param unique_labels: A 1D numpy array containing the unique integer labels present in `label_image`.
        It is assumed that labels in `label_image` are 1-based, and `unique_labels` implicitly
        covers the range of these labels, such that `label - 1` can be used as a valid index
        into an array of length `len(unique_labels)`.
    :param label_image: A 2D numpy array (mask) of integer labels, where each non-zero value
        represents a distinct region. Spots are counted only if they fall within these regions.
    :param intensity_image: A 3D numpy array with dimensions (Y, X, C) representing the
        microscopy image data, where Y and X are spatial dimensions and C is the channel dimension.
    :param model: An optional pre-initialized UFish model instance. If None, a new UFish model
        will be initialized and its weights loaded within the function.
    :param kwargs: Additional keyword arguments to be passed to `skimage.feature.peak_local_max`.
        Note that `threshold_abs` will be dynamically calculated and any `threshold_abs`
        provided in `kwargs` will be ignored with a warning. Only arguments valid for
        `peak_local_max` will be used from `kwargs`.
    :param return_spots_coords: If True, it will return the dataframe with spots coordinates.
    :return: A dictionary where keys are formatted as "Spots_Count_{ChannelName}" and values
        are 1D numpy arrays containing the spot counts for each unique label.
    """
    results: dict[str, np.ndarray] = {}
    # Initialize and load weights for the UFish model if not provided.
    if model is None:
        model = _load_ufish_model()

    if unique_labels.size == 0:
        # If unique_labels is empty, the map is trivial.
        label_to_idx_map = np.array([-1], dtype=int)
    else:
        max_label_val_in_unique = np.max(unique_labels)
        # Initialize with -1, indicating labels not present in unique_labels or invalid
        label_to_idx_map = np.full(max_label_val_in_unique + 1, -1, dtype=int)

        # Create a boolean mask for valid unique labels (positive and within max_label_val_in_unique)
        valid_unique_labels_mask = (unique_labels > 0) & (
            unique_labels <= max_label_val_in_unique
        )

        # Get the actual valid label values and their corresponding indices in the unique_labels array
        valid_unique_labels_values = unique_labels[valid_unique_labels_mask]
        valid_unique_labels_indices = np.arange(len(unique_labels))[
            valid_unique_labels_mask
        ]

        # Use advanced indexing to fill the map in a vectorized way
        label_to_idx_map[valid_unique_labels_values] = valid_unique_labels_indices

        # Log warnings for any unique_labels that were not valid (e.g., non-positive or too large)
        if not np.all(valid_unique_labels_mask):
            invalid_labels = unique_labels[~valid_unique_labels_mask]
            for label_val in np.unique(invalid_labels):  # Log unique invalid labels
                logger.warning(
                    f"Unique label {label_val} is out of expected range (1 to {max_label_val_in_unique}) or non-positive. This label will be ignored in counting."
                )
    y_list = []
    x_list = []
    channel_list = []
    for j in c:  # Iterate through specified channel indices
        # Extract the 2D image for the current channel.
        img_2d = intensity_image[..., j]
        peaks_coords, enhanced_img = model.predict(
            img_2d,
            connectivity=connectivity,
            intensity_threshold=intensity_threshold,
            laplace_process=laplace_process,
            axes=axes,
        )
        peaks_coords = peaks_coords.values
        counts_per_label = np.zeros(len(unique_labels), dtype=int)
        if len(peaks_coords) > 0:
            # Extract y and x coordinates from detected peaks and ensure they are integers for indexing.
            y_coords = peaks_coords[:, 0].astype(int)
            x_coords = peaks_coords[:, 1].astype(int)

            # Filter out points that are outside the label_image dimensions to prevent IndexError.
            valid_indices = (
                (y_coords >= 0)
                & (y_coords < label_image.shape[0])
                & (x_coords >= 0)
                & (x_coords < label_image.shape[1])
            )

            y_coords_valid = y_coords[valid_indices]
            x_coords_valid = x_coords[valid_indices]
            if len(x_coords_valid) > 0 and return_spots_coords:
                y_list.append(y_coords_valid)
                x_list.append(x_coords_valid)
                channel_list.append(np.full(x_coords_valid.shape, j, dtype=np.uint8))
            # Get the label value for each valid spot coordinate.
            labels_at_spots = label_image[y_coords_valid, x_coords_valid]

            # Filter out zero labels (background) and map to bin indices
            positive_labels_at_spots = labels_at_spots[labels_at_spots > 0]

            if len(positive_labels_at_spots) > 0:
                # Use the pre-calculated map to get the bin indices for bincount
                # Filter out any labels that were not in unique_labels (mapped to -1)
                bin_indices = label_to_idx_map[positive_labels_at_spots]
                bin_indices = bin_indices[bin_indices != -1]

                if len(bin_indices) > 0:
                    # Use bincount on these mapped indices.
                    # minlength ensures the array has at least len(unique_labels) bins.
                    binned_counts = np.bincount(
                        bin_indices, minlength=len(unique_labels)
                    )
                    # Slice to ensure the output is exactly len(unique_labels)
                    counts_per_label = binned_counts[: len(unique_labels)]
                # If bin_indices is empty after filtering, counts_per_label remains all zeros.

        # Store the counts array for the current channel.
        results[f"Spots_Count_{channel_names[j]}"] = counts_per_label
    if return_spots_coords:
        if y_list and x_list:
            final_y = np.concatenate(y_list)
            final_x = np.concatenate(x_list)
            final_channel = np.concatenate(channel_list)
        else:
            final_y, final_x, final_channel = [], [], []
        return results, pd.DataFrame(
            {"y": final_y, "x": final_x, "channel_idx": final_channel}
        )
    return results
