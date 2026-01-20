SCALLOPS_IMAGE_SCALE = "SCALLOPS_IMAGE_SCALE"
"""Set to 1 to perform the following filtered = np.clip(filtered, 0, 65535) / 65535 filtered =
skimage.img_as_uint(filtered) Additionally, includes non-SBS channels in computations to match ops
rescaling and casts corrected intensity to int."""
SCALLOPS_BASE_ORDER = "SCALLOPS_BASE_ORDER"
"""Set to comma separated list of bases for base calling when there are ties.

Set to "A,C,G,T" to match original OPS code.
"""

SCALLOPS_ALIGN_AFTER_SEGMENT = "SCALLOPS_ALIGN_AFTER_SEGMENT"
"""Set to 1 to perform segmentation before alignment."""
