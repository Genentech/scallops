"""Module containing constants for feature calculations.

Authors:
    - The SCALLOPS development team
"""

from scallops.features.colocalization import _colocalization_pairs, colocalization
from scallops.features.cp_measure_wrapper import (
    cp_granularity,
    cp_size_shape,
)
from scallops.features.intensity import intensity
from scallops.features.intensity_distribution import intensity_distribution
from scallops.features.neighbors import neighbors
from scallops.features.other import corr_region, intersects_boundary
from scallops.features.spots import spot_count
from scallops.features.texture import haralick, pftas

_cp_features_mask = {
    "sizeshape": cp_size_shape,
    "neighbors": neighbors,
}

_cp_features_single_channel = {
    "intensity": intensity,
    "granularity": cp_granularity,
    "intensitydistribution": intensity_distribution,
    "haralick": haralick,
    "pftas": pftas,
}
_other_features_single_channel = {
    "intersectsboundary": intersects_boundary,
    "spots": spot_count,
}

_cp_features_multichannel = {
    "colocalization": colocalization,
}

_other_features_multichannel = {
    "correlationpearsonbox": corr_region,
}

_features_single_channel = {}
_features_single_channel.update(_cp_features_single_channel)
_features_single_channel.update(_other_features_single_channel)
_features_multichannel = {}
_features_multichannel.update(_cp_features_multichannel)
_features_multichannel.update(_other_features_multichannel)

_features = {}
_features.update(_features_single_channel)
_features.update(_features_multichannel)

_features.update(_cp_features_mask)
_label_name_to_prefix = {"nuclei": "Nuclei", "cell": "Cells", "cytosol": "Cytoplasm"}

_features_rewrite = {
    "colocalization": _colocalization_pairs,
}

_metadata_columns_whitelist = [
    "AreaShape_Area_pheno_to_iss_qc",
    "AreaShape_BoundingBoxMaximum_X",
    "AreaShape_BoundingBoxMaximum_Y",
    "AreaShape_BoundingBoxMinimum_X",
    "AreaShape_BoundingBoxMinimum_Y",
    "AreaShape_Center_X",
    "AreaShape_Center_Y",
    "Correlation_PearsonBox",
    "Location_IntersectsBoundary",
    "Neighbors_FirstClosestObjectNumber",
    "Neighbors_SecondClosestObjectNumber",
]

_metadata_columns_whitelist_str = "|".join(_metadata_columns_whitelist)
_centroid_column_names = ["Nuclei_AreaShape_Center_Y", "Nuclei_AreaShape_Center_X"]

# see pycytominer/data/blocklist_features.txt
_blacklist = {
    "Nuclei_Correlation_Manders",
    "Nuclei_Correlation_RWC",
    "Nuclei_Granularity_14",
    "Nuclei_Granularity_15",
    "Nuclei_Granularity_16",
}

_metadata_columns_whitelist = [
    "AreaShape_BoundingBoxMaximum_X",
    "AreaShape_BoundingBoxMaximum_Y",
    "AreaShape_BoundingBoxMinimum_X",
    "AreaShape_BoundingBoxMinimum_Y",
    "AreaShape_Center_X",
    "AreaShape_Center_Y",
    "Correlation_PearsonBox",
    "Location_IntersectsBoundary",
    "Neighbors_FirstClosestObjectNumber",
    "Neighbors_SecondClosestObjectNumber",
]

_metadata_columns_whitelist_str = "|".join(_metadata_columns_whitelist)
_centroid_column_names = ["Nuclei_AreaShape_Center_Y", "Nuclei_AreaShape_Center_X"]

# see pycytominer/data/blocklist_features.txt
_blacklist = {
    "Nuclei_Correlation_Manders",
    "Nuclei_Correlation_RWC",
    "Nuclei_Granularity_14",
    "Nuclei_Granularity_15",
    "Nuclei_Granularity_16",
}
