# sphinx-build -E -q -b text docs text-out docs/features.rst
HELP = """
Available Features
******************


CellProfiler Features
=====================

* intensity
     Measures several intensity features for identified objects.
     Parameters:

     1. c: Channel index.

* granularity
     Outputs spectra of size measurements of the textures in the
     image. Parameters:

     1. c:  Channel index.

* intensity-distribution
     Measures several intensity features for identified objects.
     Parameters:

     1. c: Channel index.

     2. calculate zernike: Whether to compute Zernike moments (default
        false).

* haralick
     Measures the degree and nature of textures within objects to
     quantify their roughness and smoothness. Parameters:

     1. c: Channel index.

     2. scale: Number of pixels included in gray-level co-occurence
        matrix (Default 3).

* sizeshape
     Measures several area and shape features of identified objects.

* neighbors
     Calculates how many neighbors each object has and records various
     properties about the neighbors’ relationships, including the
     percentage of an object’s edge pixels that touch a neighbor.

* colocalization
     Measures the colocalization and correlation between intensities
     in different channels on a pixel-by-pixel basis within identified
     objects. Parameters:

     1. c1: First channel index.

     2. c2: Second channel index.


Other Features
==============

* pftas
     Parameter-free threshold adjacency statistics. Outputs 54
     features. Reference: Fast automated cell phenotype image
     classification Parameters:

     1. c: Channel index.

* correlation-pearson-box
     Pearson correlation coefficient between two channels in the label
     bounding box. Typically used to measure nuclei alignment quality
     of ISS and phenotype images. Parameters:

     1. c1: First channel index.

     2. c2: Second channel index.

* intersects-boundary
     Determines whether a label intersects a stitch boundary.
     Parameters:

     1. c: Channel index.

* spots
     Counts the number of spots in a FISH image. Parameters:

     1. c: Channel index.

     2. min peak_distance: Minimum number of pixels separating peaks
        (default 3).

     3. radius: Radius of the disk footprint used for non-maximum
        suppression in peak_local_max (default 3).


Shortcuts
=========

Use *** for all channels. Example: *intensity_**,
*colocalization_*_**.

Include a comma separated list of channel indices (0-based) to
include. Example: *intensity_0,1,2,6*.


Notes
=====

Feature names are case insensitive (intensity == Intensity) and
hyphens in feature names are ignored (intensitydistribution ==
intensity-distribution)



"""
