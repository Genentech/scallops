from collections.abc import Sequence

import numpy as np
import skimage
from mahotas.features import haralick as ma_haralick
from mahotas.features import pftas as ma_pftas
from scipy.ndimage import find_objects

F_HARALICK = """AngularSecondMoment Contrast Correlation Variance
InverseDifferenceMoment SumAverage SumVariance SumEntropy Entropy
DifferenceVariance DifferenceEntropy InfoMeas1 InfoMeas2""".split()


def pftas(
    c: Sequence[int],
    channel_names: Sequence[str],
    unique_labels: np.ndarray,
    label_image: np.ndarray,
    intensity_image: np.ndarray,
    **kwargs,
) -> dict[str, np.ndarray]:
    """Parameter-free threshold adjacency statistics. Outputs 54 features.

    Thresholding is applied using Otsu's method. Reference:
    https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-110
    """

    intensity_image = intensity_image[..., c]
    intensity_image = skimage.util.img_as_ubyte(intensity_image, force_copy=True)

    objects = find_objects(label_image)
    index = 0
    values = np.zeros((len(unique_labels), 54, len(c)))
    for object_index, sl in enumerate(objects):
        if sl is None:
            continue

        intensity_image_sl = intensity_image[sl]
        label = object_index + 1
        image = label_image[sl] == label
        intensity_image_sl = intensity_image_sl * np.expand_dims(image, -1)

        for channel_index in range(len(c)):
            values[index, :, channel_index] = ma_pftas(
                intensity_image_sl[..., channel_index]
            )

        index = index + 1
    results = {}
    for i in range(54):
        for channel_index in range(len(c)):
            results[f"Texture_PFTAS_{channel_names[c[channel_index]]}_{i}"] = values[
                :, i, channel_index
            ]
    return results


def haralick(
    c: Sequence[int],
    scale: int = 3,
    channel_names: Sequence[str] = None,
    unique_labels: np.ndarray = None,
    label_image: np.ndarray = None,
    intensity_image: np.ndarray = None,
    gray_levels: int = 256,
    ignore_zeros: bool = True,
    **kwargs,
):
    intensity_image = intensity_image[..., c]
    n_directions = 13 if intensity_image.ndim > 3 else 4

    intensity_image = skimage.util.img_as_ubyte(intensity_image, force_copy=True)
    intensity_image[~label_image.astype(bool)] = 0
    if gray_levels != 256:
        intensity_image = skimage.exposure.rescale_intensity(
            intensity_image, in_range=(0, 255), out_range=(0, gray_levels - 1)
        ).astype(np.uint8)

    values = np.zeros((len(unique_labels), n_directions, 13, len(c)))
    values[:] = np.nan
    objects = find_objects(label_image)
    index = 0

    for object_index, sl in enumerate(objects):
        if sl is None:
            continue

        intensity_image_sl = intensity_image[sl]
        label = object_index + 1
        image = label_image[sl] == label
        intensity_image_sl = intensity_image_sl * np.expand_dims(image, -1)
        for channel_index in range(len(c)):
            try:
                # (n_directions, 13)
                values[index, ..., channel_index] = ma_haralick(
                    intensity_image_sl[..., channel_index],
                    distance=scale,
                    ignore_zeros=ignore_zeros,
                )

            except ValueError:
                pass
        index = index + 1

    results = {}
    for channel_index in range(len(c)):
        for direction in range(n_directions):
            for feature_index, feature_name in enumerate(F_HARALICK):
                results[
                    "Texture_{}_{}_{:d}_{:02d}_{:d}".format(
                        feature_name,
                        channel_names[c[channel_index]],
                        scale,
                        direction,
                        gray_levels,
                    )
                ] = values[..., direction, feature_index, channel_index]
    return results
