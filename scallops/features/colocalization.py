"""
MeasureColocalization
=====================

**MeasureColocalization** measures the colocalization and correlation
between intensities in different images (e.g., different color channels)
on a pixel-by-pixel basis, within identified objects or across an entire
image.

Given two or more images, this module calculates the correlation &
colocalization (Overlap, Manders, Costes’ Automated Threshold & Rank
Weighted Colocalization) between the pixel intensities. The correlation
/ colocalization can be measured for entire images, or a correlation
measurement can be made within each individual object. Correlations /
Colocalizations will be calculated between all pairs of images that are
selected in the module, as well as between selected objects. For
example, if correlations are to be measured for a set of red, green, and
blue images containing identified nuclei, measurements will be made
between the following:

-  The blue and green, red and green, and red and blue images.
-  The nuclei in each of the above image pairs.

A good primer on colocalization theory can be found on the `SVI website`_.

You can find a helpful review on colocalization from Aaron *et al*. `here`_.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  *Correlation:* The correlation between a pair of images *I* and *J*,
   calculated as Pearson’s correlation coefficient. The formula is
   covariance( *I* , *J*)/[std( *I* ) × std( *J*)].
-  *Slope:* The slope of the least-squares regression between a pair of
   images I and J. Calculated using the model *A* × *I* + *B* = *J*, where *A* is the slope.
-  *Overlap coefficient:* The overlap coefficient is a modification of
   Pearson’s correlation where average intensity values of the pixels are
   not subtracted from the original intensity values. For a pair of
   images R and G, the overlap coefficient is measured as r = sum(Ri *
   Gi) / sqrt (sum(Ri*Ri)*sum(Gi*Gi)).
-  *Manders coefficient:* The Manders coefficient for a pair of images R
   and G is measured as M1 = sum(Ri_coloc)/sum(Ri) and M2 =
   sum(Gi_coloc)/sum(Gi), where Ri_coloc = Ri when Gi > 0, 0 otherwise
   and Gi_coloc = Gi when Ri >0, 0 otherwise.
-  *Manders coefficient (Costes Automated Threshold):* Costes’ automated
   threshold estimates maximum threshold of intensity for each image
   based on correlation. Manders coefficient is applied on thresholded
   images as Ri_coloc = Ri when Gi > Gthr and Gi_coloc = Gi when Ri >
   Rthr where Gthr and Rthr are thresholds calculated using Costes’
   automated threshold method.
-  *Rank Weighted Colocalization coefficient:* The RWC coefficient for a
   pair of images R and G is measured as RWC1 =
   sum(Ri_coloc*Wi)/sum(Ri) and RWC2 = sum(Gi_coloc*Wi)/sum(Gi),
   where Wi is Weight defined as Wi = (Rmax - Di)/Rmax where Rmax is the
   maximum of Ranks among R and G based on the max intensity, and Di =
   abs(Rank(Ri) - Rank(Gi)) (absolute difference in ranks between R and
   G) and Ri_coloc = Ri when Gi > 0, 0 otherwise and Gi_coloc = Gi
   when Ri >0, 0 otherwise. (Singan et al. 2011, BMC Bioinformatics
   12:407).

References
^^^^^^^^^^

-  Aaron JS, Taylor AB, Chew TL. Image co-localization - co-occurrence versus correlation.
   J Cell Sci. 2018;131(3):jcs211847. Published 2018 Feb 8. doi:10.1242/jcs.211847



.. _SVI website: http://svi.nl/ColocalizationTheory
.. _here: https://jcs.biologists.org/content/joces/131/3/jcs211847.full.pdf
"""

from collections.abc import Sequence

import numpy as np
import scipy.ndimage
import scipy.stats
from scipy.linalg import lstsq
from scipy.ndimage import find_objects

# Modified Correlation-Pearson
F_CORRELATION_FORMAT = "Correlation_Pearson"

"""Feature name format for the slope measurement"""
F_SLOPE_FORMAT = "Correlation_Slope"

"""Feature name format for the overlap coefficient measurement"""
F_OVERLAP_FORMAT = "Correlation_Overlap"

"""Feature name format for the Manders Coefficient measurement"""
F_K_FORMAT = "Correlation_K"

"""Feature name format for the Manders Coefficient measurement"""
F_KS_FORMAT = "Correlation_KS"

"""Feature name format for the Manders Coefficient measurement"""
F_MANDERS_FORMAT = "Correlation_Manders"

"""Feature name format for the RWC Coefficient measurement"""
F_RWC_FORMAT = "Correlation_RWC"

"""Feature name format for the Costes Coefficient measurement"""
F_COSTES_FORMAT = "Correlation_Costes"

"""
thr : int or float, optional
    Set threshold as percentage of maximum intensity for the images (default 15).
    You may choose to measure colocalization metrics only for those pixels above
    a certain threshold. Select the threshold as a percentage of the maximum intensity
    of the above image [0-99].
    This value is used by the Overlap, Manders, and Rank Weighted Colocalization
    measurements.

fast_costes : {M_FASTER, M_FAST, M_ACCURATE}, optional
    Method for Costes thresholding (default M_FASTER).
    This setting determines the method used to calculate the threshold for use within the
    Costes calculations. The *{M_FAST}* and *{M_ACCURATE}* modes will test candidate thresholds
    in descending order until the optimal threshold is reached. Selecting *{M_FAST}* will attempt
    to skip candidates when results are far from the optimal value being sought. Selecting *{M_ACCURATE}*
    will test every possible threshold value. When working with 16-bit images these methods can be extremely
    time-consuming. Selecting *{M_FASTER}* will use a modified bisection algorithm to find the threshold
    using a shrinking window of candidates. This is substantially faster but may produce slightly lower
    thresholds in exceptional circumstances.
    In the vast majority of instances the results of all strategies should be identical. We recommend using
    *{M_FAST}* mode when working with 8-bit images and *{M_FASTER}* mode when using 16-bit images.
    Alternatively, you may want to disable these specific measurements entirely
    (available when "*Run All Metrics?*" is set to "*No*").
"""


def _pearson_pair(x1, x2):
    corr = np.corrcoef(x1, x2)[0, 1]
    slope = lstsq(np.array((x1, np.ones_like(x1))).transpose(), x2)[0][0]
    return (corr, slope)


def _correlation_overlap_pair(x1, x2, thr):
    threshold1 = (thr / 100) * x1.max()
    threshold2 = (thr / 100) * x2.max()
    mask = (x1 >= threshold1) & (x2 >= threshold2)
    x1 = x1[mask]
    x2 = x2[mask]
    fpsq = np.sum(x1**2)
    spsq = np.sum(x2**2)
    pdt = np.sqrt(fpsq * spsq)
    overlap = np.nan if pdt == 0 else np.sum(x1 * x2) / pdt
    k1 = np.nan if fpsq == 0 else np.sum(x1 * x2) / fpsq
    k2 = np.nan if spsq == 0 else np.sum(x1 * x2) / spsq
    return (overlap, k1, k2)


def _correlation_costes_pair(x1, x2, scale):
    first_pixels = x1
    second_pixels = x2
    thr_fi_c, thr_si_c = _bisection_costes(first_pixels, second_pixels, scale)

    # Costes' threshold for entire image is applied to each object
    fi_above_thr = first_pixels > thr_fi_c
    si_above_thr = second_pixels > thr_si_c
    combined_thresh_c = fi_above_thr & si_above_thr
    fi_thresh_c = first_pixels[combined_thresh_c]
    si_thresh_c = second_pixels[combined_thresh_c]

    tot_fi_thr_c = first_pixels[first_pixels >= thr_fi_c].sum()
    tot_si_thr_c = second_pixels[second_pixels >= thr_si_c].sum()
    c1 = 0 if tot_fi_thr_c == 0 else fi_thresh_c.sum() / tot_fi_thr_c
    c2 = 0 if tot_si_thr_c == 0 else si_thresh_c.sum() / tot_si_thr_c
    return (c1, c2)


def _correlation_rwc_pair(x1, x2, thr):
    first_pixels = x1
    second_pixels = x2
    Rank1 = np.lexsort([first_pixels])
    Rank2 = np.lexsort([second_pixels])
    Rank1_U = np.hstack([[False], first_pixels[Rank1[:-1]] != first_pixels[Rank1[1:]]])
    Rank2_U = np.hstack(
        [[False], second_pixels[Rank2[:-1]] != second_pixels[Rank2[1:]]]
    )
    Rank1_S = np.cumsum(Rank1_U)
    Rank2_S = np.cumsum(Rank2_U)
    Rank_im1 = np.zeros(first_pixels.shape, dtype=int)
    Rank_im2 = np.zeros(second_pixels.shape, dtype=int)
    Rank_im1[Rank1] = Rank1_S
    Rank_im2[Rank2] = Rank2_S

    R = max(Rank_im1.max(), Rank_im2.max()) + 1
    Di = abs(Rank_im1 - Rank_im2)
    weight = ((R - Di) * 1.0) / R

    tff = (thr / 100) * first_pixels.max()
    tss = (thr / 100) * second_pixels.max()
    combined_thresh = (first_pixels >= tff) & (second_pixels >= tss)
    fi_thresh = first_pixels[combined_thresh]
    si_thresh = second_pixels[combined_thresh]
    tot_fi_thr = (first_pixels[first_pixels >= tff]).sum()
    tot_si_thr = (second_pixels[second_pixels >= tss]).sum()
    weight_thresh = weight[combined_thresh]

    RWC1 = (fi_thresh * weight_thresh).sum() / tot_fi_thr
    RWC2 = (si_thresh * weight_thresh).sum() / tot_si_thr
    return (RWC1, RWC2)


def _manders_fold_pair(x1, x2, thr):
    first_pixels = x1
    second_pixels = x2
    tff = (thr / 100) * first_pixels.max()
    tss = (thr / 100) * second_pixels.max()
    combined_thresh = (first_pixels >= tff) & (second_pixels >= tss)
    fi_thresh = first_pixels[combined_thresh]
    si_thresh = second_pixels[combined_thresh]
    tot_fi_thr = first_pixels[first_pixels >= tff].sum()
    tot_si_thr = second_pixels[second_pixels >= tss].sum()

    M1 = fi_thresh.sum() / tot_fi_thr
    M2 = si_thresh.sum() / tot_si_thr

    return (M1, M2)


def _bisection_costes(fi: np.ndarray, si: np.ndarray, scale_max: int = 255):
    """
    Finds the Costes Automatic Threshold for colocalization using a bisection algorithm.
    Candidate thresholds are selected from within a window of possible intensities,
    this window is narrowed based on the R value of each tested candidate.
    We're looking for the first point below 0, and R value can become highly variable
    at lower thresholds in some samples. Therefore the candidate tested in each
    loop is 1/6th of the window size below the maximum value (as opposed to the midpoint).
    """

    non_zero = (fi > 0) | (si > 0)
    xvar = np.var(fi[non_zero], axis=0, ddof=1)
    yvar = np.var(si[non_zero], axis=0, ddof=1)

    xmean = np.mean(fi[non_zero], axis=0)
    ymean = np.mean(si[non_zero], axis=0)

    z = fi[non_zero] + si[non_zero]
    zvar = np.var(z, axis=0, ddof=1)

    covar = 0.5 * (zvar - (xvar + yvar))

    denom = 2 * covar
    num = (yvar - xvar) + np.sqrt((yvar - xvar) * (yvar - xvar) + 4 * (covar * covar))
    a = num / denom
    b = ymean - a * xmean

    # Initialise variables
    left = 1
    right = scale_max
    mid = ((right - left) // (6 / 5)) + left
    lastmid = 0
    # Marks the value with the last positive R value.
    valid = 1

    while lastmid != mid:
        thr_fi_c = mid / scale_max
        thr_si_c = (a * thr_fi_c) + b
        combt = (fi < thr_fi_c) | (si < thr_si_c)
        if np.count_nonzero(combt) <= 2:
            # Can't run pearson with only 2 values.
            left = mid - 1
        else:
            try:
                costReg, _ = scipy.stats.pearsonr(fi[combt], si[combt])
                if costReg < 0:
                    left = mid - 1
                elif costReg >= 0:
                    right = mid + 1
                    valid = mid
            except ValueError:
                # Catch misc Pearson errors with low sample numbers
                left = mid - 1
        lastmid = mid
        if right - left > 6:
            mid = ((right - left) // (6 / 5)) + left
        else:
            mid = ((right - left) // 2) + left

    thr_fi_c = (valid - 1) / scale_max
    thr_si_c = (a * thr_fi_c) + b

    return thr_fi_c, thr_si_c


# MODIFIED: This reproduces the behaviour of the block at
#  https://github.com/cellprofiler/CellProfiler/blob/450abdc2eaa0332cb6d1d4aaed4bf0a4b843368d/src/subpackages/core/cellprofiler_core/image/abstract_image/file/_file_image.py#L396-L405
def infer_scale(data: np.ndarray) -> int:
    if data.dtype in [np.int8, np.uint8]:
        scale = 255
    elif data.dtype in [np.int16, np.uint16]:
        scale = 65535
    elif data.dtype == np.int32:
        scale = 2**32 - 1
    elif data.dtype == np.uint32:
        scale = 2**32
    else:
        scale = 1

    return scale


def colocalization(
    c1: int,
    c2: int,
    channel_names: Sequence[str],
    unique_labels: np.ndarray,
    label_image: np.ndarray,
    intensity_image: np.ndarray,
) -> dict[str, np.ndarray]:
    pass


def _colocalization_pairs(
    c: list[tuple[int, int]],
    channel_names: Sequence[str],
    unique_labels: np.ndarray,
    label_image: np.ndarray,
    intensity_image: np.ndarray,
    threshold=15,
    **kwargs,
) -> dict[str, np.ndarray]:
    objects = find_objects(label_image)
    index = 0
    n_pairs = len(c)
    values = np.zeros((len(unique_labels), n_pairs, 11))
    scale = infer_scale(intensity_image)
    for object_index, sl in enumerate(objects):
        if sl is None:
            continue

        label = object_index + 1
        image = label_image[sl] == label
        intensity_image_sl = intensity_image[sl][image]

        for pair_index in range(n_pairs):
            pair = c[pair_index]
            x1 = intensity_image_sl[..., pair[0]]
            x2 = intensity_image_sl[..., pair[1]]
            corr, slope = _pearson_pair(x1, x2)
            values[index, pair_index, 0] = corr
            values[index, pair_index, 1] = slope
            m1, m2 = _manders_fold_pair(x1, x2, threshold)
            values[index, pair_index, 2] = m1
            values[index, pair_index, 3] = m2

            overlap, k1, k2 = _correlation_overlap_pair(x1, x2, threshold)
            values[index, pair_index, 4] = overlap
            values[index, pair_index, 5] = k1
            values[index, pair_index, 6] = k2

            RWC1, RWC2 = _correlation_rwc_pair(x1, x2, threshold)
            values[index, pair_index, 7] = RWC1
            values[index, pair_index, 8] = RWC2

            c1, c2 = _correlation_costes_pair(x1, x2, scale)
            values[index, pair_index, 9] = c1
            values[index, pair_index, 10] = c2
        index = index + 1
    results = {}
    for pair_index in range(n_pairs):
        pair = c[pair_index]
        channel_name1 = channel_names[pair[0]]
        channel_name2 = channel_names[pair[1]]
        results[f"{F_CORRELATION_FORMAT}_{channel_name1}_{channel_name2}"] = values[
            :, pair_index, 0
        ]
        results[f"{F_SLOPE_FORMAT}_{channel_name1}_{channel_name2}"] = values[
            :, pair_index, 1
        ]
        results[f"{F_MANDERS_FORMAT}_{channel_name1}_{channel_name2}"] = values[
            :, pair_index, 2
        ]
        results[f"{F_MANDERS_FORMAT}_{channel_name2}_{channel_name1}"] = values[
            :, pair_index, 3
        ]
        results[f"{F_OVERLAP_FORMAT}_{channel_name1}_{channel_name2}"] = values[
            :, pair_index, 4
        ]
        results[f"{F_K_FORMAT}_{channel_name1}_{channel_name2}"] = values[
            :, pair_index, 5
        ]
        results[f"{F_K_FORMAT}_{channel_name2}_{channel_name1}"] = values[
            :, pair_index, 6
        ]

        results[f"{F_RWC_FORMAT}_{channel_name1}_{channel_name2}"] = values[
            :, pair_index, 7
        ]
        results[f"{F_RWC_FORMAT}_{channel_name2}_{channel_name1}"] = values[
            :, pair_index, 8
        ]
        results[f"{F_COSTES_FORMAT}_1"] = values[:, pair_index, 9]
        results[f"{F_COSTES_FORMAT}_2"] = values[:, pair_index, 10]
    return results
