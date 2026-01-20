"""Segmentation Module.

Submodules for various segmentation algorithms and utility functions.

Submodules:
    - cellpose: Cellpose Segmentation submodule.
    - stardist: Stardist Segmentation submodule.
    - watershed: Watershed Segmentation submodule.
    - util: Segmentation Utility submodule.


Authors:
    - The SCALLOPS development team
"""

from .util import (  # noqa: F401
    cyto_channel_summary,
    remove_boundary_labels,
    remove_labels_by_area,
)
