Command Line
***************

Each step of an Optical Pooled Screen (OPS) analysis is a ``scallops`` command. This page lists every
command and its options; see :doc:`example_commands` for the commands in pipeline order and
:ref:`outputs` for the files they write.

scallops dialout
================

The ``scallops dialout`` command analyzes sequencing data from pooled dial-out libraries: it counts
guides, aligns reads to a reference, and reports the results. It has two subcommands:

analysis:
    Reads FASTQ files, aligns reads to the reference with BWA, matches sequences to the library
    design by Hamming distance, and writes per-pool counts, dropouts, summary statistics and
    (optionally) unaligned reads.

report:
    Combines the ``analysis`` results of several pools into plots of summary statistics, matches,
    mismatches and guide ranks, written to a PDF or to PNG files.

.. argparse::
   :module: scallops.__main__
   :func: create_parsers
   :prog: scallops
   :path: dialout


scallops extract-crops
======================

The ``scallops extract-crops`` command extracts image crops from labeled images.

.. argparse::
   :module: scallops.__main__
   :func: create_parsers
   :prog: scallops
   :path: extract-crops

scallops find-objects
======================

The ``scallops find-objects`` command finds objects in label images output from segmentation.

.. argparse::
   :module: scallops.__main__
   :func: create_parsers
   :prog: scallops
   :path: find-objects


scallops features
=================

The ``scallops features`` command is used to compute various features from labeled images, producing output
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

.. include:: ../scallops/features/features.rst


scallops illum-corr
====================

The ``scallops illum-corr agg`` command computes a flat-field image that corrects uneven illumination,
which would otherwise bias segmentation and intensity features. Tiles are aggregated with the mean,
median or min (``--agg-method``), smoothed with a median filter (radius set with ``--smooth``; ``0``
disables it) and optionally rescaled. The output is written as OME-TIFF or OME-Zarr. ``scallops stitch``
applies the flat-field with ``--ffp``.

.. argparse::
   :module: scallops.__main__
   :func: create_parsers
   :prog: scallops
   :path: illum-corr agg


scallops pooled-sbs
===================

The ``scallops pooled-sbs`` command processes in situ sequencing-by-synthesis (SBS) images: it detects
spots, calls reads and assigns barcodes to cells, and merges the barcodes with phenotype data.

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


scallops pert-map
===================

The ``scallops pert-map`` command builds and evaluates perturbation maps from per-cell features (for
example the ``merge-features`` output of the OPS workflow). It replaces the older ``norm-features``
and ``rank-features`` commands. Each subcommand reads one or more Zarr, h5ad or Parquet inputs and
writes to Zarr, h5ad or Parquet.

Key Features:

* **filter**: Drop labels and features before downstream processing, by query/Parquet filters
  (``--label-filter``, ``--feature-filter``) and by feature variance (``--min-feature-variance``,
  ``--max-feature-variance``, ``--n-features``).
* **normalize**: Z-score (or local z-score) normalize features, optionally relative to a reference
  population (``--reference-query``, e.g. non-targeting controls) and stratified by group (``--by``).
* **rank**: Compare each perturbation against a reference (``--reference``) using Welch's t-test,
  Student's t-test, or Mann-Whitney U, producing per-feature effect sizes and FDR-corrected p-values.
* **pca**: Reduce features to principal components, fit on a reference population if given.
* **tvn**: Typical variation normalization — further align treatments and controls across batches
  using the covariance of reference (control) observations.
* **aggregate**: Aggregate single-cell data to one row per perturbation (``--by``), optionally
  centering to a reference first.
* **similarity-matrix**: Compute a pairwise similarity matrix between aggregated perturbations.
* **enrichment**: Test similarity-matrix neighborhoods for enrichment of gene sets (``--set``, GMT
  format).
* **recall**: Evaluate a similarity matrix against known relationships (``--ground-truth-corum``) at
  given recall thresholds.

See :doc:`the Perturbation Map Workflow <workflows>` for how these steps compose into an end-to-end
pipeline, and :ref:`example_commands:Rank Features` for a worked CLI example.

.. argparse::
   :module: scallops.__main__
   :func: create_parsers
   :prog: scallops
   :path: pert-map



scallops registration
======================

The ``scallops registration`` command provides functionality for performing image registration.
This includes image alignment using elastix (ITK) and applying precomputed transformations to images or labels.

Key Features:

* **ITK Registration**: The elastix subcommand performs registration of moving images to fixed images or across timepoints
  using elastix (ITK). It supports the use of pre-configured ITK parameters and outputs transformed images and/or
  labels in Zarr format.
* **Transform Application**: The ``transformix`` subcommand applies previously computed ITK transformations to images or labels,
  storing the results in a Zarr output directory.

.. argparse::
   :module: scallops.__main__
   :func: create_parsers
   :prog: scallops
   :path: registration

.. include:: registration_opts.rst

scallops segment
================

The ``scallops segment`` command provides a command-line interface (CLI) for performing nuclei and cell
segmentation. It supports various segmentation algorithms and outputs segmented labels in Zarr format.

Key Features:

* **Nuclei Segmentation**: The ``nuclei`` subcommand performs nuclei segmentation using StarDist (default) or Cellpose, with
  optional filtering based on area and intensity.
* **Cell Segmentation**: The ``cell`` subcommand performs cell segmentation using propagation (default), watershed,
  watershed-intensity, or Cellpose. It also supports various cytoplasmic channels, thresholding, and post-segmentation
  filtering.
* **Overlap**: The ``overlap`` subcommand finds the fraction of each label in one segmentation that is contained within
  labels of another segmentation (``--meta-output``), then renumbers the second segmentation (and any additional
  labels such as cytosol) to match the overlapping label in the first, writing the result to ``--label-output``. This
  matters most for cell segmentation algorithms (e.g. Cellpose) whose cell and nuclei labels don't already share IDs;
  with watershed or propagation, where they already do, renumbering has no effect.

.. argparse::
   :module: scallops.__main__
   :func: create_parsers
   :prog: scallops
   :path: segment


scallops stitch
================

The ``scallops stitch`` command provides a command-line interface (CLI) for performing stitching of
microscopy images.

Key Features:

* **Performance**: Use Dask for parallel processing.
* **Cross-correlation**: Run cross-correlation with and without phase normalization and keep whichever gives the
  better result.
* **Stage Position Handling**: Read stage positions directly from Bioformats-supported images, such as ``.nd2`` files,
  or from a CSV or Parquet file.
* **Comprehensive Output**: Outputs stitched image in OME-ZARR format, stitched positions in Parquet format,
  PDF report, tile boundary mask, and tile source labels in OME-ZARR format, plus a per-image channel alignment
  record when channel alignment correction is used.
* **Z-index**: Select a Z index, a maximum Z projection, the best-focused Z index of each tile, or per-tile Z
  indices from a Parquet file.
* **Blending**: Enable or disable image blending during stitching. When not blending, use tile closest to well center
  in overlapping regions.
* **Crop**: Crop image tiles to remove edge effects, both when aligning tiles and when writing the stitched image; set
  automatically when radial correction is enabled.
* **Radial Correction**: Automatically determine K for radial distortion and apply radial correction.
* **Channel Alignment Within Tiles**: Align channels to a reference channel by a translation estimated separately in
  each tile (``--channel-reference``).
* **Channel Alignment Correction**: Align channels to the reference channel (magnification and translation per channel)
  using values from a parameter file or estimated from the tiles of each well (see :ref:`channel_alignment`).
* **Stitching Evaluation**: Compute error in overlapping regions after stitching.

.. argparse::
   :module: scallops.__main__
   :func: create_parsers
   :prog: scallops
   :path: stitch


.. _channel_alignment:

Channel alignment correction
----------------------------

Chromatic aberration and other optical differences between filter sets can make a channel slightly magnified and
shifted relative to the others. Because the offset comes from the optics, it is the same in every tile. Use
``--alignment-correction-channels`` to correct it when writing the stitched image: each listed channel is aligned
to the reference channel (``--align-channel``) with a magnification ``m`` about the tile center and a translation
``ty``, ``tx`` in pixels::

    corrected(y, x) = raw(c + (1 + m) * ((y, x) - c) + (ty, tx)),   c = tile center

Pass channel indices (0-based), or ``all`` for every output channel except the reference. Listed indices that are
the reference channel or are not in ``--output-channels`` are skipped with a warning. ``--alignment-correction-channel-order`` is
required: it gives the parameter-file name of every input channel, in input order (use ``-`` for a channel without
values). A warning is logged if a name differs from the channel name in the image metadata.

``--alignment-correction-channels`` cannot be combined with ``--channel-reference``. ``--channel-reference``
estimates a translation only, separately in each tile, by cross-correlation; use it when channel offsets vary from
tile to tile. Use ``--alignment-correction-channels`` for a fixed offset, especially one that includes a
magnification difference.

Parameter file
^^^^^^^^^^^^^^

Values are read from a JSON parameter file (``--alignment-correction-params``), usually one per microscope and
objective. Example::

    {
      "format": "scallops-channel-alignment",
      "version": 1,
      "tile_shape": [2304, 2304],
      "reference_channel": "DAPI",
      "channels": {
        "FITC": {"m": 0.000751, "ty": 0.341, "tx": -0.162},
        "TRITC": {"same_as": "FITC"},
        "CY7": {"m": -0.001493, "ty": 0.950, "tx": 0.300}
      }
    }

* ``format`` and ``version`` must be as shown.
* ``tile_shape`` is the tile ``[height, width]`` in pixels that the values were measured on; stitching stops with
  an error if the tiles have a different shape.
* ``reference_channel`` (optional) is the name of the channel the values are relative to; stitching stops with an
  error if it differs from the ``--alignment-correction-channel-order`` name of ``--align-channel``.
* ``channels`` maps each channel name to its ``m``, ``ty``, and ``tx``, or to ``same_as`` naming another channel
  whose values it reuses. Names are matched to ``--alignment-correction-channel-order`` ignoring case.

To build a parameter file, stitch without one so that the values are estimated (see below), then copy the ``m``,
``ty``, and ``tx`` of each channel from the per-image record into the file.

Estimation
^^^^^^^^^^

Requested channels without values in the parameter file (or all of them, if no file is given) are estimated from
the tiles of each stitched image (e.g. a well). The reference channel must be a nuclear stain such as DAPI: nuclei
are found on it, and each channel's offset is measured from its average image around the nuclei. Up to 100 valid
tiles are sampled at random across the four quadrants of the well (interior tiles only for wells with more than
100 tiles), and the median of the per-tile values is used. An estimate fails its quality checks when:

* fewer than 20 tiles, or fewer than half of the tiles tried, give a valid estimate;
* the 95% bootstrap uncertainty of the correction at the tile corners is more than 0.5 px; or
* the correction at the tile corners is more than 10 px, which is implausible.

If any estimate fails, stitching stops with an error naming the channel and the reason. Give values for that
channel in the parameter file or leave it out of ``--alignment-correction-channels``.

Resampling and output
^^^^^^^^^^^^^^^^^^^^^

The alignment is combined with the radial correction into one coordinate map per channel, so each channel is
resampled only once. When the correction is requested, every output channel, corrected or not, is resampled with a
quintic B-spline instead of the default bilinear radial correction, which keeps more fine detail.

The values applied to each image, where they came from (parameter file or estimate), and the estimation statistics
are saved to ``<image>-channel-alignment.json`` in ``--report-output`` and to the stitched image metadata.

Edge bands
^^^^^^^^^^

Correcting a channel leaves a band of missing pixels along some tile edges (pixels that would come from outside the
tile). The radial correction leaves such a band in every channel on its own; only bands wider than that count.

* Along an axis with neighboring tiles, the crop SCALLOPS applies when writing the stitched image removes bands up
  to its width. A wider band would need a larger crop, which could open gaps between tiles, so stitching stops with
  an error: leave that channel out of ``--alignment-correction-channels`` or give smaller values for it (which may
  correct it only partly).
* Along an axis without neighboring tiles (e.g. a single tile), the crop is widened to remove the band.
* If no tile pairs overlap enough to be aligned, tiles are placed at their stage positions and nothing is cropped,
  so stitching stops with an error naming the channels that would leave a band; leave those out of
  ``--alignment-correction-channels``. (When tile pairs overlap but every alignment is rejected, tiles are also
  placed at their stage positions, but the crop still applies.)

The crop used is saved in the per-image record (``fusion_crop``).

.. _bioformats: https://www.openmicroscopy.org/bio-formats/
.. _OME-ZARR: https://github.com/ome/ome-zarr-py
.. _skimage: https://scikit-image.org/


scallops stitch-preview
=======================

The ``scallops stitch-preview`` command provides a quick preview of stitched multi-tile microscopy images, allowing
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
