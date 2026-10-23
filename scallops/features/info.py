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
     Measures radial intensity distribution features for identified objects.
     Parameters:

     1. c: Channel index.

     2. bins: Number of bins to measure the distribution (default 4).


* intensity-distribution-zernike
     Measures zernike intensity distribution features for identified objects.
     Parameters:

     1. c: Channel index.

     2. moment: Maximum zernike moment (default 9)



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

* puncta
     Detects puncta with an adaptive threshold (pixel > alpha * local
     background) followed by a watershed split on a highpass gaussian
     filtered image, and measures puncta counts, areas, intensities,
     and spatial distribution. Watershed pieces that share one
     saturated (flat-topped) plateau are rejoined into a single
     punctum. Outputs 33 features per channel, including the quality
     flag FlagNoNuclearCentroid. Spatial features are measured from
     the centroid of each label's nucleus (the nuclei label with the
     same ID, read from the "-nuclei" labels in "--labels", used only
     if it overlaps or touches the label); a label without such a
     nucleus uses its own centroid and gets FlagNoNuclearCentroid = 1.
     If fewer than half of the labels sampled in an image overlap or
     touch a nucleus with the same ID (e.g. cellpose cells, which are
     numbered independently of the nuclei), the nuclei are not used
     for that image, every label gets FlagNoNuclearCentroid = 1, and a
     warning is logged. Parameters are encoded in the feature names
     (e.g. "Puncta_PunctaCount_Channel3_s10.0_a1.5_m2_d2").
     Parameters:

     1. c: Channel index.

     2. sigma: Gaussian sigma for calculating local background used
        for foreground calculation and for high-pass filter; 0 skips
        the filter and uses the ROI mean as the background (default
        10).

     3. alpha: Threshold multiplier on the local background; must be
        >= 1 when sigma > 0 (default 1.5).

     4. min_size_fg_objects: Minimum puncta area in pixels; smaller
        foreground objects are discarded; smaller puncta after
        watershed are merged (default 2).

     5. peak_min_distance: Minimum distance in pixels between
        watershed peaks; larger values merge nearby peaks into one
        punctum (default 2).


Shortcuts
=========

Use *** for all channels. Example: *intensity_**,
*colocalization_*_**.

Include a comma separated list of channel indices (0-based) to
include. Example: *intensity_0,1,2,6*.

Specify a range of channel indices using start:stop:step.
Example: *colocalization_0_1:10:2*.

Notes
=====

Feature names are case insensitive (intensity == Intensity) and
hyphens in feature names are ignored (intensitydistribution ==
intensity-distribution)



"""
