Available Features
-------------------

CellProfiler Features
^^^^^^^^^^^^^^^^^^^^^^^^


* `intensity <https://cellprofiler-manual.s3.amazonaws.com/CellProfiler-4.2.8/modules/measurement.html#measureobjectintensity>`_
    Measures several intensity features for identified objects. Parameters:

    #. c: Channel index.


* `granularity <https://cellprofiler-manual.s3.amazonaws.com/CellProfiler-4.2.8/modules/measurement.html#measuregranularity>`_
    Outputs spectra of size measurements of the textures in the image. Parameters:

    #. c:  Channel index.

* `intensity-distribution <https://cellprofiler-manual.s3.amazonaws.com/CellProfiler-4.2.8/modules/measurement.html#measureobjectintensitydistribution>`_
    Measures radial intensity features for identified objects. Parameters:

    #. c: Channel index.
    #. bins: Number of bins to measure the distribution (default 4)

* `intensity-distribution-zernike <https://cellprofiler-manual.s3.amazonaws.com/CellProfiler-4.2.8/modules/measurement.html#measureobjectintensitydistribution>`_
    Measures zernike intensity features for identified objects. Parameters:

    #. c: Channel index.
    #. moment: Maximum zernike moment (default 9)

* `haralick <https://cellprofiler-manual.s3.amazonaws.com/CellProfiler-4.2.8/modules/measurement.html#measuretexture>`_
    Measures the degree and nature of textures within objects to quantify their roughness and smoothness. Parameters:

    #. c: Channel index.
    #. scale: Number of pixels included in gray-level co-occurence matrix (Default 3).


* `sizeshape <https://cellprofiler-manual.s3.amazonaws.com/CellProfiler-4.2.8/modules/measurement.html#measureobjectsizeshape>`_
    Measures several area and shape features of identified objects.

* `neighbors <https://cellprofiler-manual.s3.amazonaws.com/CellProfiler-4.2.8/modules/measurement.html#measureobjectneighbors>`_
    Calculates how many neighbors each object has and records various properties about the neighbors’ relationships,
    including the percentage of an object’s edge pixels that touch a neighbor.

* `colocalization <https://cellprofiler-manual.s3.amazonaws.com/CellProfiler-4.2.8/modules/measurement.html#measurecolocalization>`_
    Measures the colocalization and correlation between intensities in different channels on a pixel-by-pixel
    basis within identified objects. Parameters:

    #. c1: First channel index.
    #. c2: Second channel index.



Other Features
^^^^^^^^^^^^^^^^

* pftas
    Parameter-free threshold adjacency statistics. Outputs 54 features.
    Reference: `Fast automated cell phenotype image classification <https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-110>`_
    Parameters:

    #. c: Channel index.

* correlation-pearson-box
    Pearson correlation coefficient between two channels in the label bounding box. Typically used
    to measure nuclei alignment quality of ISS and phenotype images. Parameters:

    #. c1: First channel index.
    #. c2: Second channel index.

* intersects-boundary
    Determines whether a label intersects a stitch boundary. Parameters:

    #. c: Channel index.

* spots
    Counts the number of spots in a FISH image. Parameters:

    #. c: Channel index.
    #. min peak_distance: Minimum number of pixels separating peaks (default 3).
    #. radius: Radius of the disk footprint used for non-maximum suppression in peak_local_max (default 3).

* puncta
    Detects puncta with an adaptive threshold (pixel > alpha * local background) followed by a
    watershed split on a highpass gaussian filtered image, and measures puncta counts, areas, intensities, and spatial distribution.
    Watershed pieces that share one saturated (flat-topped) plateau are rejoined into a single
    punctum.
    Outputs 33 features per channel, including the quality flag FlagNoNuclearCentroid. Spatial
    features are measured from the centroid of each label's nucleus (the nuclei label with the
    same ID, read from the ``-nuclei`` labels in ``--labels``, used only if it overlaps or
    touches the label); a label without such a nucleus uses its own centroid and gets
    FlagNoNuclearCentroid = 1. Parameters are encoded in the feature names
    (e.g. ``Puncta_PunctaCount_Channel3_s10.0_a1.5_m2_d2``).

    Nuclei labels: for each image, puncta expects the nuclei labels (``<image>-nuclei``) in the
    same labels zarr as the ROI labels passed to ``--labels``, with IDs matching the cells, as
    produced by propagation/watershed segmentation or by ``scallops segment overlap``. If they
    are not found there, puncta still runs: it logs a warning, measures the spatial features
    from each ROI's own centroid, and sets FlagNoNuclearCentroid = 1 for every ROI. Note that
    ``scallops segment overlap`` writes its renumbered cells to a separate zarr without the
    nuclei, so cell and cytoplasm features run on that zarr currently use this fallback.
    Parameters:

    #. c: Channel index.
    #. sigma: Gaussian sigma for calculating local background used for foreground calculation and for high-pass filter; 0 skips the filter and uses the ROI mean as the background (default 10).
    #. alpha: Threshold multiplier on the local background; must be >= 1 when sigma > 0 (default 1.5).
    #. min_size_fg_objects: Minimum puncta area in pixels; smaller foreground objects are discarded; smaller puncta after watershed are merged (default 2).
    #. peak_min_distance: Minimum distance in pixels between watershed peaks; larger values merge
       nearby peaks into one punctum (default 2).


.. _shortcuts:

Shortcuts
^^^^^^^^^^^

Use `*` for all channels. Example: `intensity_*`, `colocalization_*_*`.

Include a comma separated list of channel indices (0-based) to include. Example: `intensity_0,1,2,6`.

Specify a range of channel indices using start:stop:step. Example: *colocalization_0_1:10:2*.

Notes
^^^^^^
Feature names are case insensitive (intensity == Intensity) and hyphens in feature names are
ignored (intensitydistribution == intensity-distribution)
