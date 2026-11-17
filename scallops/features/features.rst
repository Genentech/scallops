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
    Counts the number of spots in a FISH image using U-FISH_. Parameters:

    #. c: Channel index.
    #. intensity threshold: The intensity threshold (default 0.06).
    #. connectivity: Connectivity for local maxima (default 2).


* puncta
    Detects puncta with an adaptive threshold (pixel > alpha * local background) followed by a
    watershed split on a highpass gaussian filtered image, and measures puncta counts, areas, intensities, and spatial distribution.
    Watershed pieces that share one saturated (flat-topped) plateau are rejoined into a single
    punctum.
    Outputs 33 features per channel, including the quality flag FlagNoNuclearCentroid. With the
    default anchor ``nc``, spatial features are measured from the centroid of each label's
    nucleus (the nuclei label with the same ID, read from the ``-nuclei`` labels in
    ``--labels``, used only if it overlaps or touches the label); a label without such a
    nucleus uses its own centroid and gets FlagNoNuclearCentroid = 1. With anchor ``rc``, every
    label uses its own centroid (FlagNoNuclearCentroid = 1). Parameters are encoded in the
    feature names (e.g. ``Puncta_PunctaCount_Channel3_s10.0_a1.5_m2_d2_nc``).

    When to use: puncta is intended for punctate and blobular cellular structures in
    immunofluorescence phenotyping images (e.g. vesicles, granules, Golgi fragments, receptor
    clusters), which vary in size and are often elongated, and for when their size, intensity,
    and position in the cell matter, not just their number. It needs no model or GPU. For
    counting diffraction-limited FISH spots, use ``spots`` (U-FISH), which is designed for that
    case; puncta has not been validated on FISH data. In our tests on immunofluorescence
    images, U-FISH did not detect these structures well.

    Nuclei labels: for each image, puncta looks for the nuclei labels (``<image>-nuclei``) in the
    ``--labels`` stores, with IDs matching the cells, as produced by propagation/watershed
    segmentation or by ``scallops segment overlap``. ``scallops segment overlap`` writes its
    renumbered cells to a separate store without the nuclei, so for those cells pass both
    stores, e.g. ``--labels segment.zarr segment-overlap.zarr`` (for labels present in both, the
    store listed last is used). With anchor ``nc``, ``scallops features`` stops with an error if
    the nuclei labels are not found; use anchor ``rc`` when no nuclei segmentation exists.
    Parameters:

    #. c: Channel index.
    #. sigma: Gaussian sigma for calculating local background used for foreground calculation and for high-pass filter; 0 skips the filter and uses the ROI mean as the background (default 10).
    #. alpha: Threshold multiplier on the local background; must be >= 1 when sigma > 0 (default 1.5).
    #. min_size_fg_objects: Minimum puncta area in pixels; smaller foreground objects are discarded; smaller puncta after watershed are merged (default 2).
    #. peak_min_distance: Minimum distance in pixels between watershed peaks; larger values merge
       nearby peaks into one punctum (default 2).
    #. anchor: Reference point of the spatial features: ``nc`` (nucleus centroid) or ``rc``
       (ROI centroid) (default ``nc``).


.. _shortcuts:

Feature Name Syntax
^^^^^^^^^^^^^^^^^^^

A feature is written as its name followed by its parameters, separated by underscores and given in
the order listed above: ``name_param1_param2``. For example, ``intensity_0`` measures intensity in
channel 0, and ``colocalization_0_1`` measures colocalization between channels 0 and 1. Features
without parameters, such as ``sizeshape``, are written as the name alone.

Trailing parameters can be left out to use their defaults. For example, ``intensitydistribution_0``
uses 4 bins and ``intensitydistribution_0_8`` uses 8; ``puncta_3`` uses the default sigma, alpha,
minimum size, peak distance and anchor, while ``puncta_3_5_2`` sets sigma to 5 and alpha to 2. A
parameter can only be set if every parameter before it is also given.

Channel parameters are 0-based and accept several shortcuts:

* All channels: ``*``, for example ``intensity_*`` or ``colocalization_*_*``.
* A list of channels: comma-separated, for example ``intensity_0,1,2,6``.
* A range: ``start:stop:step`` (``stop`` is excluded), for example ``colocalization_0_1:10:2``.

When ``--stack-images`` is given, prefix a channel index with ``s`` to refer to a channel of the
stacked image instead of the primary image, for example ``correlationpearsonbox_0_s0`` (channel 0
of ``--images`` against channel 0 of ``--stack-images``). The ``s`` prefix takes a single channel
index, not a list or range.

Feature names are case insensitive (``intensity`` is the same as ``Intensity``), and hyphens in
feature names are ignored (``intensitydistribution`` is the same as ``intensity-distribution``).


.. _U-FISH:  https://u-fish.readthedocs.io/en/latest/index.html
