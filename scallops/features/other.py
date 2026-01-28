import logging
from collections.abc import Sequence

import numpy as np
from skimage.measure import regionprops

logger = logging.getLogger("scallops")


def intersects_boundary(
    c: Sequence[int],
    channel_names: Sequence[str],
    label_image: np.ndarray,
    intensity_image: np.ndarray,
    **kwargs,
) -> dict[str, np.ndarray]:
    """Determine whether a label intersects a stitch boundary.

    :param c: Channels
    :param channel_names: Channel names
    :param label_image: Label image (output from segmentation)
    :param intensity_image: Masked labels typically from stitch.zarr/labels with image
        pattern {plate}-{well}-mask) where zeros indicate locations where tiles overlap
    """
    result = {}
    props = regionprops(label_image, intensity_image=intensity_image)

    for channel in c:
        values = np.zeros(len(props), dtype=bool)
        for index in range(len(props)):
            r = props[index]
            image = r.image.astype(np.uint8)
            mask = r.image_intensity[..., channel] * image
            values[index] = np.any(mask != image)
        channel_name = channel_names[channel]
        result[f"Location_IntersectsBoundary_{channel_name}"] = values
    return result


def corr_region(
    c1: int,
    c2: int,
    unique_labels: np.ndarray,
    channel_names: Sequence[str],
    label_image: np.ndarray,
    intensity_image: np.ndarray,
    **kwargs,
) -> dict:
    """Calculate the Pearson correlation coefficient between two channels using all
    values in region bounding box.
    """

    props = regionprops(label_image, intensity_image=intensity_image)
    assert len(props) == len(unique_labels)
    values = np.zeros(len(props))
    for index in range(len(props)):
        r = props[index]
        img_slice = r.image_intensity
        x1 = img_slice[..., c1].flatten()
        x2 = img_slice[..., c2].flatten()
        corr = np.corrcoef((x1, x2))[1, 0]
        values[index] = corr

    results = {}
    results[f"Correlation_PearsonBox_{channel_names[c1]}_{channel_names[c2]}"] = values
    return results
