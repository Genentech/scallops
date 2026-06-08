Tutorials and Examples
==========================

The following notebooks walk through key SCALLOPS workflows, from getting familiar with the
library's data model to running a complete end-to-end Optical Pooled Screen (OPS) analysis.
They are designed to be followed in order for new users, though each notebook can also be
consulted independently as a reference.

All notebooks can be run locally with a working SCALLOPS installation (see :doc:`install`).
Example data paths in the notebooks may need to be updated to reflect your local setup.

----

End-to-End Tutorial
--------------------

A production-grade walkthrough of the full SCALLOPS pipeline using the PERISCOPE dataset.
Covers every major stage in sequence: illumination correction, image stitching, ISS
registration, cellular segmentation, spot detection, base calling, feature extraction, and
final data merging. This is the recommended starting point for understanding how all
components fit together in a real experiment. Downstream analyses, quality control, and
visualization are explored in more depth in the `OPS Analysis`_ and `Plotting`_ notebooks.

`Open notebook <notebooks/End-to-End.ipynb>`_

----

OPS Analysis
-------------

A focused walkthrough of the Optical Pooled Screening analysis steps: ISS image registration,
nuclei and cell segmentation, spot detection and base calling, barcode assignment to cells, and
phenotypic feature extraction. Includes visualization and quality-control metrics throughout.
Useful as a reference for any individual OPS analysis step.

`Open notebook <notebooks/ops.ipynb>`_

----

Plotting
---------

A showcase of SCALLOPS' built-in visualization tools. Covers composite image plotting,
cross-talk analysis, distribution plots, heatmaps, histograms, and image display utilities.
Also demonstrates Napari-based interactive exploration for OPS data. Useful as a quick
reference when building figures or inspecting results.

`Open notebook <notebooks/plotting.ipynb>`_

----

Data Structures
----------------

An introduction to SCALLOPS' core data model. Demonstrates how to read individual images and
full experiments, concatenate image collections across dimensions, apply functions across an
experiment, and save and reload data using the Zarr format. Useful for understanding how
SCALLOPS represents and manipulates imaging data.

`Open notebook <notebooks/data_structures.ipynb>`_

----

Barrel Distortion Estimation
------------------------------

A tutorial on identifying and correcting optical barrel distortion in microscopy images.
Explains the mathematical model behind barrel distortion and walks through the interactive
workflow in Napari for estimating and applying the correction coefficient. Relevant for
users working with wide-field objectives or lenses that introduce radial distortion.

`Open notebook <notebooks/barrel_distortion_estimation.ipynb>`_
