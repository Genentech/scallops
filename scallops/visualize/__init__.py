"""SCALLOPS Visualization Module.

This module provides submodules for visualization tasks related to biological imaging data.
The submodules include tools for composite image generation, crosstalk analysis, distribution plotting,
heatmap visualization, histogram plotting, image display, integration with Napari, image registration,
segmentation visualization, and utility functions.

Submodules:
- composite: Composite image visualization.
- crosstalk: Visualization of crosstalk between channels.
- distribution: Distribution plotting for various analyses.
- heatmap: Heatmap visualization of data.
- histogram: Histogram plotting for channel and barcode analysis.
- imshow: General image display utilities.
- napari: Integration with the Napari viewer for interactive exploration.
- registration: Visualization of image registration results.
- segmentation: Visualization of segmentation results.
- utils: General utility functions for visualization tasks.
"""

from .composite import (  # noqa: F401
    experiment_composite,
    imcomposite,
    label_montage,
    montage_plot,
)
from .crosstalk import pairwise_channel_scatter_plot  # noqa: F401
from .distribution import (  # noqa: F401
    cdf_plot,
    ridge_plot,
    volcano_plot,
)
from .heatmap import (  # noqa: F401
    base_call_mismatches_heatmap,
    in_situ_identity_matrix_plot,
    plate_heatmap,
)
from .histogram import channel_hist_plot, in_situ_barcode_hist_plot  # noqa: F401
