# Adapted from https://github.com/afermg/cp_measure/blob/main/src/cp_measure/core/measurecolocalization.py


from collections.abc import Sequence

import numpy as np
from cp_measure.core.measurecolocalization import (
    F_CORRELATION_FORMAT,
    F_COSTES_FORMAT,
    F_K_FORMAT,
    F_MANDERS_FORMAT,
    F_OVERLAP_FORMAT,
    F_RWC_FORMAT,
    F_SLOPE_FORMAT,
    M_FASTER,
    _costes_pair,
    _manders_pair,
    _overlap_pair,
    _pearson_pair,
    _rwc_pair,
    infer_scale,
)
from scipy.ndimage import find_objects


def colocalization(
    c1: int,
    c2: int,
    channel_names: Sequence[str],
    unique_labels: np.ndarray,
    label_image: np.ndarray,
    intensity_image: np.ndarray,
) -> dict[str, np.ndarray]:
    pass


def _colocalization_pairs(
    c: list[tuple[int, int]],
    channel_names: Sequence[str],
    unique_labels: np.ndarray,
    label_image: np.ndarray,
    intensity_image: np.ndarray,
    threshold=15,
    fast_costes: str = M_FASTER,
    **kwargs,
) -> dict[str, np.ndarray]:
    objects = find_objects(label_image)
    index = 0
    n_pairs = len(c)
    values = np.zeros((len(unique_labels), n_pairs, 11))
    scale = infer_scale(intensity_image)
    for object_index, sl in enumerate(objects):
        if sl is None:
            continue

        label = object_index + 1
        image = label_image[sl] == label
        intensity_image_sl = intensity_image[sl][image]

        for pair_index in range(n_pairs):
            pair = c[pair_index]
            x1 = intensity_image_sl[..., pair[0]]
            x2 = intensity_image_sl[..., pair[1]]
            corr, slope = _pearson_pair(x1, x2)
            values[index, pair_index, 0] = corr
            values[index, pair_index, 1] = slope
            m1, m2 = _manders_pair(x1, x2, threshold)
            values[index, pair_index, 2] = m1
            values[index, pair_index, 3] = m2

            overlap, k1, k2 = _overlap_pair(x1, x2, threshold)
            values[index, pair_index, 4] = overlap
            values[index, pair_index, 5] = k1
            values[index, pair_index, 6] = k2

            RWC1, RWC2 = _rwc_pair(x1, x2, threshold)
            values[index, pair_index, 7] = RWC1
            values[index, pair_index, 8] = RWC2

            c1, c2 = _costes_pair(x1, x2, scale, fast_costes)
            values[index, pair_index, 9] = c1
            values[index, pair_index, 10] = c2
        index = index + 1
    results = {}
    for pair_index in range(n_pairs):
        pair = c[pair_index]
        channel_name1 = channel_names[pair[0]]
        channel_name2 = channel_names[pair[1]]
        results[f"{F_CORRELATION_FORMAT}_{channel_name1}_{channel_name2}"] = values[
            :, pair_index, 0
        ]
        results[f"{F_SLOPE_FORMAT}_{channel_name1}_{channel_name2}"] = values[
            :, pair_index, 1
        ]
        results[f"{F_MANDERS_FORMAT}_{channel_name1}_{channel_name2}"] = values[
            :, pair_index, 2
        ]
        results[f"{F_MANDERS_FORMAT}_{channel_name2}_{channel_name1}"] = values[
            :, pair_index, 3
        ]
        results[f"{F_OVERLAP_FORMAT}_{channel_name1}_{channel_name2}"] = values[
            :, pair_index, 4
        ]
        results[f"{F_K_FORMAT}_{channel_name1}_{channel_name2}"] = values[
            :, pair_index, 5
        ]
        results[f"{F_K_FORMAT}_{channel_name2}_{channel_name1}"] = values[
            :, pair_index, 6
        ]

        results[f"{F_RWC_FORMAT}_{channel_name1}_{channel_name2}"] = values[
            :, pair_index, 7
        ]
        results[f"{F_RWC_FORMAT}_{channel_name2}_{channel_name1}"] = values[
            :, pair_index, 8
        ]
        results[f"{F_COSTES_FORMAT}_{channel_name1}_{channel_name2}"] = values[
            :, pair_index, 9
        ]
        results[f"{F_COSTES_FORMAT}_{channel_name2}_{channel_name1}"] = values[
            :, pair_index, 10
        ]
    return results
