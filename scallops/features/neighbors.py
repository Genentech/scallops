import numpy
from cp_measure.multimask.measureobjectneighbors import D_EXPAND

from scallops.features.cp_measure_wrapper import cp_neighbors
from scallops.segmentation.util import relabel_sequential


# Identical to cp_measure but removes labels that are not part of dask block so neighbors for labels that span
# multiple blocks are not output more than once"
def neighbors(
    unique_labels_original: numpy.ndarray,
    label_image_original: numpy.ndarray,
    distance: int = 5,
    distance_method: str = D_EXPAND,
    **kwargs,
) -> dict[str, numpy.ndarray]:
    unique_labels = numpy.unique(label_image_original)
    unique_labels = unique_labels[unique_labels > 0]
    label_filter = numpy.isin(unique_labels, unique_labels_original, assume_unique=True)
    label_image_original = relabel_sequential(label_image_original, unique_labels)
    result = cp_neighbors(label_image_original, distance, distance_method)

    renumber_cols = [
        "Neighbors_FirstClosestObjectNumber_Expanded",
        "Neighbors_SecondClosestObjectNumber_Expanded",
    ]
    for key in result.keys():
        val = result[key].flatten()
        if key in renumber_cols:
            val = unique_labels[val - 1]
        val = val[label_filter]
        result[key] = val
    return result
