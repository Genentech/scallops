"""Module for frameshift analysis.

This module provides functionalities to analyze and process frameshift data.
The functions in this module help to identify, quantify, and visualize frameshift events
in biological sequences.

Authors:
    - The SCALLOPS development team
"""

import logging

import numpy as np
import pandas as pd
import xarray as xr
from skimage.measure import regionprops_table
from sklearn.feature_selection import mutual_info_regression
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from scallops._demux import compute_threshold, extract_peaks, process_spots
from scallops.reads import (
    apply_channel_crosstalk_matrix,
    assign_barcodes_to_labels,
    channel_crosstalk_matrix,
    decode_max,
)
from scallops.registration.crosscorrelation import align_images
from scallops.segmentation.stardist import segment_nuclei_stardist
from scallops.segmentation.watershed import segment_cells_watershed

logger = logging.getLogger("scallops")


def classify_epitope(
    image: np.ndarray, label: np.ndarray, th_prob: float = 0.9
) -> pd.DataFrame:
    """Classifies an epitope based on input features.

    :param: image: np.ndarray
        The image data containing the epitope.
    :param: label: np.ndarray
        The labeled regions corresponding to the image.
    :param: th_prob: float
        Threshold probability for classifying the epitope (default is 0.9).
    :return: pd.DataFrame
        DataFrame containing the classification results.

    :example:
    .. code-block:: python

        # Example usage of classify_epitope
        result = classify_epitope(image, label, th_prob=0.9)
    """

    rps = regionprops_table(label, image, properties=("intensity_mean", "label"))
    exclude = rps["intensity_mean"] == 0
    rps["intensity_mean"] = rps["intensity_mean"][~exclude]
    rps["label"] = rps["label"][~exclude]
    v = np.log(rps["intensity_mean"])
    gm = GaussianMixture(n_components=2).fit(np.expand_dims(v, 1))
    prob = gm.predict_proba(np.expand_dims(v, 1))
    ind = (prob[:, 0] > th_prob) | (prob[:, 1] > th_prob)  # throw away uncertain cells

    pred = gm.predict(np.expand_dims(v, 1))[ind]
    if (gm.means_[0] > gm.means_[1])[0]:
        pred = 1 - pred

    df = pd.DataFrame(
        index=rps["label"][ind],
        data=dict(epitope=pred, log_intensity=v[ind]),
    )
    df.index.name = "label"
    return df


def peaks_to_bases(
    spots: np.ndarray, maxed: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """Converts peak data to base sequence positions.

    :param: spots: np.ndarray
        The spots data representing peak intensities.
    :param: maxed: np.ndarray
        The maximum intensities used for thresholding peaks.
    :param: labels: np.ndarray
        Labels corresponding to different regions in the data.
    :return: np.ndarray
        The base sequence positions derived from peak data.

    :example:
    .. code-block:: python

        # Example usage of peaks_to_bases
        result = peaks_to_bases(spots, maxed, labels)
    """
    read_mask = (spots > 0).astype(bool)

    positions = np.array(np.where(read_mask))
    # (t,c,read)
    maxed_spots = maxed.data[:, :, read_mask]

    coords = pd.DataFrame(
        data=dict(y=positions[0], x=positions[1], peak=spots[read_mask])
    )
    if labels is not None:
        coords["label"] = labels[read_mask]
    data = xr.DataArray(
        maxed_spots,
        dims=["t", "c", "read"],
        coords=dict(
            read=pd.MultiIndex.from_frame(coords), c=maxed.c.values, t=maxed.t.values
        ),
    ).transpose(*("read", "t", "c"))
    return data


def detect_barcode(
    image: xr.DataArray,
    label: np.ndarray,
    channels: list[int] = [1, 2, 3, 4],
    distance: int = 50,
    n_smallest_clusters: int = 3,
) -> xr.DataArray:
    """Detects barcode positions within the image data.

    :param: image: xr.DataArray
        The image data containing the barcode.
    :param: label: np.ndarray
        The labeled regions corresponding to the image.
    :param: channels: List[int]
        Channels used for barcode detection (default is [1, 2, 3, 4]).
    :param: distance: int
        Distance parameter for peak detection (default is 50).
    :param: n_smallest_clusters: int
        Number of smallest clusters to consider in the threshold calculation (default is 3).
    :return: xr.DataArray
        An xarray DataArray containing the detected barcode positions.

    :example:
    .. code-block:: python

        # Example usage of detect_barcode
        result = detect_barcode(
            image, label, channels=[1, 2, 3, 4], score_method="softmax"
        )
    """
    pixel_array, cell_labels = extract_peaks(
        image.isel(t=0, c=channels).data.copy(), label, distance
    )
    thresholds = compute_threshold(
        pixel_array,
        n_clusters=len(channels) + 1,
        n_smallest_clusters=n_smallest_clusters,
    )
    logger.info(f"thresholds are {thresholds}")
    spots = process_spots(
        [
            image.isel(t=0, c=channels).values,
        ],
        [
            label,
        ],
        thresholds,
    )[0]
    bases_array = peaks_to_bases(spots, image.isel(c=channels), label)
    bases_array = bases_array.assign_coords(c=["G", "T", "A", "C"])  # match bases
    bases_array = bases_array.sel(
        c=["A", "C", "G", "T"]
    )  # match order for median correction ties

    w = channel_crosstalk_matrix(bases_array, method="median")
    cr = apply_channel_crosstalk_matrix(bases_array, w).squeeze()
    try:  # TODO handle NaN
        maxmi = np.triu(
            [
                mutual_info_regression(cr[:, np.delete(range(4), i)], cr[:, i])
                for i in range(4)
            ],
            k=1,
        ).max()
    except:  # noqa: E722
        maxmi = np.nan
    corrected_bases_array = apply_channel_crosstalk_matrix(bases_array, w)
    corrected_bases_array = corrected_bases_array.astype(int)

    scaler = StandardScaler()
    yy = scaler.fit_transform(corrected_bases_array[:, 0, :])
    corrected_bases_array.values = xr.DataArray(
        np.expand_dims(yy, 1), dims=["read", "t", "c"]
    )
    df_reads = decode_max(corrected_bases_array)
    df_reads = df_reads.drop(["y", "x", "peak", "label"], axis=1).reset_index()
    df_cells = assign_barcodes_to_labels(df_reads)
    return maxmi, corrected_bases_array, df_reads, df_cells


def assess_frameshift(
    phenoimage: xr.DataArray,
    issimage: xr.DataArray,
    epitope_channel: int = 1,
    iss_channels: list[int] = [1, 2, 3, 4],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assesses frameshift events in the provided sequence data.

    :param: phenoimage: xr.DataArray
        The phenotype image data used for frameshift assessment.
    :param: issimage: xr.DataArray
        The image data corresponding to in situ sequencing (ISS).
    :param: epitope_channel: int
        The channel corresponding to the epitope (default is 1).
    :param: iss_channels: List[int]
        Channels used in the ISS image (default is [1, 2, 3, 4]).
    :return: Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing two DataFrames: the first for the frameshift results and the second for the ISS results.

    :example:
    .. code-block:: python

        # Example usage of assess_frameshift
        result = assess_frameshift(
            phenoimage, issimage, epitope_channel=1, iss_channels=[1, 2, 3, 4]
        )
    """
    phenoimage = phenoimage.squeeze()
    if "z" in issimage.dims:
        issimage = issimage.squeeze("z")

    phenoimage = align_images(
        issimage.isel(t=0), phenoimage, upsample_factor=1, autoscale=False
    )

    nuclei = segment_nuclei_stardist(issimage)

    cells = segment_cells_watershed(
        issimage, nuclei, threshold="quantile", quantile_threshold=0.5
    )[0]
    epitope_df = classify_epitope(phenoimage.isel(c=epitope_channel).data, nuclei)

    maxmi, corrected_bases_array, df_reads, df_labels = detect_barcode(
        issimage, cells, channels=iss_channels
    )

    #  df_labels = df_labels.query("barcode_count_0/barcode_count > 0.5")
    return df_reads, epitope_df.join(df_labels.set_index("label")).rename(
        {"barcode_0": "barcode"}, axis=1
    )


def sensitivity_precision(
    df: pd.DataFrame,
    control_barcodes: list[str],
    targeting_barcodes: list[str],
) -> pd.Series:
    """Computes the sensitivity and precision of the frameshift analysis.

    :param: df: pd.DataFrame
        The DataFrame containing the analysis results, including barcode classifications.
    :param: control_barcodes: List[str]
        A list of control barcodes used as a reference for sensitivity and precision calculations.
    :param: targeting_barcodes: List[str]
        A list of targeting barcodes to assess the frameshift analysis accuracy.
    :return: pd.Series
        A Series containing the computed sensitivity and precision values.

    :example:
    .. code-block:: python

        # Example usage of sensitivity_precision
        sensitivity, precision = sensitivity_precision(
            df, control_barcodes, targeting_barcodes
        )
    """
    epi = df["epitope"]
    bas = df["barcode"]

    fn = ((epi == 1) & ((bas.isin(control_barcodes)) | bas.isna())).sum()  # FN
    fn_mistakes_only = ((epi == 1) & (bas.isin(control_barcodes))).sum()  # FN
    tp = ((epi == 1) & (bas.isin(targeting_barcodes))).sum()  # TP
    fp = ((epi == 0) & (bas.isin(targeting_barcodes))).sum()  # FP
    #   tn = ((epi == 0) & (bas.isin(control_barcodes)) | bas.isna()).sum()  # TN
    # accu = 1- 2*y/(x+y)

    # f1 = 2 * x /(2*x + y + fp)
    # recall, prec = x/(x+fp), x/(x+y)
    # frac, accu = recall, prec
    # accu = accu if accu > 0 else np.nan  # this value can sometimes be negative
    precision = tp / (tp + fp)
    sensitivity = tp / (tp + fn)
    sensitivity_mistakes_only = tp / (tp + fn_mistakes_only)
    return pd.Series(
        dict(
            tp=tp,
            fn=fn,
            fn_mistakes_only=fn_mistakes_only,
            fp=fp,
            sensitivity=sensitivity,
            sensitivity_mistakes_only=sensitivity_mistakes_only,
            precision=precision,
        )
    )
