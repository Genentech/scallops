Command Line
***************

SCALLOPS provides a powerful command-line interface (CLI) to automate various workflows and tasks related
to Optical Pooled Screens (OPS). Each command supports multiple arguments and options, allowing users to
customize their workflows.

scallops dialout
================

The `scallops dialout` command is designed for the analysis and reporting of pooled dialout
library sequencing data. It allows users to run the dialout pipeline to analyze sequencing data,
generate counts, align reads to a reference genome, and produce detailed reports. It provides
two subcommands:

analysis:
    The `analysis` subcommand is responsible for running the main dialout library sequencing data
    analysis pipeline. It processes sequencing reads from FASTQ files, performs alignment to a
    reference genome using BWA, computes Hamming distances between sequences, and generates output
    files such as mapped counts, dropout data, and unaligned sequences. This subcommand is essential
    for the initial data processing of a pooled dialout experiment.

report:
    The `report` subcommand generates a comprehensive report based on the analysis pipeline’s
    output. It creates visual summaries, scatter plots, and statistics from the dialout analysis
    data, helping researchers interpret sequencing results. The report can include analysis of
    guide RNA sequences, mismatches, and dropout rates, providing valuable insights into the
    sequencing performance and accuracy.

.. argparse::
   :module: scallops.__main__
   :func: create_parsers
   :prog: scallops
   :path: dialout


scallops find-objects
======================

The `scallops find-objects` command finds objects in label images output from segmentation.

.. argparse::
   :module: scallops.__main__
   :func: create_parsers
   :prog: scallops
   :path: find-objects


scallops features
=================

The `scallops features` command is used to compute various features from labeled images, producing output
in Parquet format. Each feature is indexed by the label in the corresponding image. This command allows
users to extract multiple types of features, including geometric, texture, and intensity-based features,
for each region of interest, such as nuclei, cells, or cytosol.

Key Features:

* **Multi-region Feature Computation**: Compute features for different regions in the images, such as nuclei, cells, and cytosol.
  The feature extraction process is customizable, allowing users to define which features to compute for each region.
* **Customizable Feature Sets**: Users can specify which features to compute, with available shortcuts to quickly select groups of
  features (See :ref:`shortcuts`).
* **Stacked Image Support**: SCALLOPS supports processing both primary and stacked images, allowing users to compute features
  across multiple channels by stacking different image types together.
* **Scalable via Dask**: The computation process leverages Dask for distributed and parallelized processing, enabling SCALLOPS
  to handle large image datasets efficiently.

.. argparse::
   :module: scallops.__main__
   :func: create_parsers
   :prog: scallops
   :path: features

.. include:: features.rst


scallops illum-corr
====================

The `scallops illum-corr` command is used for performing illumination correction on images, a crucial
preprocessing step in biomedical image analysis. Uneven illumination can introduce artifacts that affect
the accuracy of downstream analysis tasks like segmentation and feature extraction.
SCALLOPS provides an aggregation method for illumination correction with two aggregators: Median and mean.

Key Features:

* This method computes illumination correction by aggregating images using mean or median, followed by
  an optional median filter and rescaling. It offers a simple and effective approach for addressing
  illumination variations. The output can be saved as Zarr or TIFF images.
* This method is designed to improve image uniformity, thereby enhancing the reliability of image analysis
  workflows, particularly in the context of high-throughput biomedical imaging data.

.. argparse::
   :module: scallops.__main__
   :func: create_parsers
   :prog: scallops
   :path: illum-corr agg


scallops pooled-sbs
===================

The `scallops pooled-sbs` command is designed for processing images from pooled in-situ sequencing (SBS)
experiments. This pipeline includes spot detection, read calling, and merging of single-cell sequencing
(SCS) data with phenotype data.

Key Features:

* **Spot Detection**: The spot detection subcommand identifies candidate peaks in the image data, which correspond to
  sequencing spots. The results can be saved in multiple formats, including the raw data, filtered
  images, and detected peaks.
* **Reads Processing**: The reads subcommand processes the detected spots to assign sequencing reads to specific labels, such as
  nuclei or cells. It also includes options for crosstalk correction between channels and outputs
  corrected and uncorrected base intensities.
* **Merging Data**: The merge subcommand joins in-situ barcodes with phenotype data, allowing for a combined view of
  sequencing and phenotype information.

.. argparse::
   :module: scallops.__main__
   :func: create_parsers
   :prog: scallops
   :path: pooled-sbs


scallops norm-features
=======================

The `scallops norm-features` command is used to normalize features.

.. argparse::
   :module: scallops.__main__
   :func: create_parsers
   :prog: scallops
   :path: rank-features


scallops rank-features
=======================

The `scallops rank-features` command is used to compute significance from the output of `scallops norm-features`.

.. argparse::
   :module: scallops.__main__
   :func: create_parsers
   :prog: scallops
   :path: rank-features


scallops registration
======================

The `scallops registration` command provides functionality for performing image registration.
This includes image alignment using ITK, cross-correlation-based registration, and applying precomputed transformations
to images or labels.

Key Features:

* **ITK Registration**: The elastix subcommand performs registration of moving images to fixed images or across timepoints
  using ITK. It supports the use of pre-configured ITK parameters and outputs transformed images and/or
  labels in Zarr format.
* **Cross-Correlation Registration**: The `cross-correlation` subcommand registers images by aligning them within and across timepoints,
  using cross-correlation and specified channels for registration.
* **Transform Application**: The `transformix` subcommand applies previously computed ITK transformations to images or labels,
  storing the results in a Zarr output directory.

.. argparse::
   :module: scallops.__main__
   :func: create_parsers
   :prog: scallops
   :path: registration

.. include:: registration_opts.rst

scallops segment
================

The `scallops segment` command provides a command-line interface (CLI) for performing nuclei and cell
segmentation. It supports various segmentation algorithms and outputs segmented labels in Zarr format.

Key Features:

* **Nuclei Segmentation**: The `nuclei` subcommand performs nuclei segmentation using methods such as Stardist and Cellpose, with
  optional filtering based on area.
* **Cell Segmentation**: The `cell` subcommand performs cell segmentation using methods like Watershed, Cellpose, and
  Propagation. It also supports various cytoplasmic channels, thresholding, and post-segmentation filtering.

.. argparse::
   :module: scallops.__main__
   :func: create_parsers
   :prog: scallops
   :path: segment


scallops stitch
================

The `scallops stitch` command provides a command-line interface (CLI) for performing stitching of
microscopy images.

Key Features:

* **Performance**: Utilize dask for parallel processing.
* **Cross-correlation**: Use both phase and no normalization in cross-correlation computations, automatically choosing
  the one that gives the best result.
* **Stage Position Handling**: Read stage positions directly from Bioformats-supported images, such as `.nd2` files,
  or from a CSV file.
* **Comprehensive Output**: Outputs stitched image in OME-ZARR format, stitched positions in Parquet format,
  PDF report, tile boundary mask, and tile source labels in OME-ZARR format.
* **Z-index**: Option to specify specific Z index or perform maximum Z projection.
* **Blending**: Enable or disable image blending during stitching. When not blending, use tile closest to well center
  in overlapping regions.
* **Crop**: Crop image tiles to remove edge effects.
* **Radial Correction**: Automatically determine K for radial distortion and apply radial correction.
* **Stitching Evaluation**: Compute error in overlapping regions after stitching.

.. argparse::
   :module: scallops.__main__
   :func: create_parsers
   :prog: scallops
   :path: stitch


.. _bioformats: https://www.openmicroscopy.org/bio-formats/
.. _OME-ZARR: https://github.com/ome/ome-zarr-py
.. _skimage: https://scikit-image.org/


stitch-preview
==============

The `stitch-preview` command provides a quick preview of stitched multi-tile microscopy images, allowing
users to visualize the result before performing full stitching. It uses the stage positions for stitching
and saves the resulting image.

Key Features:

* **Tile Positioning**: Stitching uses stage positions and has options to display tile numbers and bounds.
* **Downsampling**: Enables downsampling of the image resolution to improve performance and reduce memory requirements.
* **Channel Selection**: Allows users to specify the channel to display from multi-channel images.
* **Log Transformation**: Optionally apply log transformation to pixel intensities for better visualization of dim images.

.. argparse::
   :module: scallops.__main__
   :func: create_parsers
   :prog: scallops
   :path: stitch-preview


.. include:: outputs.rst


.. _CorrectIlluminationCalculate: https://cellprofiler-manual.s3.amazonaws.com/CPmanual/CorrectIlluminationCalculate.html
.. _bioio: https://bioio-devs.github.io/bioio/
.. _dask: https://www.dask.org/
.. _`(Singh et al. J Microscopy, 2014)`: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4359755/
.. _`Blainey lab code`: https://github.com/feldman4/OpticalPooledScreens
.. _elastix: https://elastix.lumc.nl/
