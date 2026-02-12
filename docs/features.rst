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


.. _shortcuts:

Shortcuts
^^^^^^^^^^^

Use `*` for all channels. Example: `intensity_*`, `colocalization_*_*`.

Include a comma separated list of channel indices (0-based) to include. Example: `intensity_0,1,2,6`.

Notes
^^^^^^
Feature names are case insensitive (intensity == Intensity) and hyphens in feature names are
ignored (intensitydistribution == intensity-distribution)
