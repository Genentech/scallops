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
     Counts the number of spots in a FISH image using U-FISH.
     Parameters:

     1. c: Channel index.

     2. intensity threshold: The intensity threshold (default 0.06).

     3. connectivity: Connectivity for local maxima (default 2).

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
     Parameters are encoded in the feature names (e.g.
     "Puncta_PunctaCount_Channel3_s10.0_a1.5_m2_d2").

     When to use: puncta is intended for punctate and blobular
     cellular structures in immunofluorescence phenotyping images
     (e.g. vesicles, granules, Golgi fragments, receptor clusters),
     which vary in size and are often elongated, and for when their
     size, intensity, and position in the cell matter, not just their
     number. It needs no model or GPU. For counting diffraction-
     limited FISH spots, use "spots" (U-FISH), which is designed for
     that case; puncta has not been validated on FISH data. In our
     tests on immunofluorescence images, U-FISH did not detect these
     structures well.

     Nuclei labels: for each image, puncta looks for the nuclei labels
     ("<image>-nuclei") in the "--labels" stores, with IDs matching
     the cells, as produced by propagation/watershed segmentation or
     by "scallops segment overlap". "scallops segment overlap" writes
     its renumbered cells to a separate store without the nuclei, so
     for those cells pass both stores, e.g. "--labels segment.zarr
     segment-overlap.zarr" (for labels present in both, the store
     listed last is used). If the nuclei labels are not found, puncta
     still runs: it logs a warning, measures the spatial features from
     each ROI's own centroid, and sets FlagNoNuclearCentroid = 1 for
     every ROI. Parameters:

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
