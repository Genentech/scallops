import logging

import numpy as np
import scipy.stats
import statsmodels.api as sm
from scipy.stats import gmean

from scallops.spots import _find_peaks

logger = logging.getLogger("scallops")


def clr_2d(matrix: np.ndarray) -> np.ndarray:
    """Perform centered log-ratio (CLR) transformation on a 2D matrix.

    This function replaces zero values with the smallest non-zero value in the same column to avoid
    issues with logarithms. It then calculates the geometric mean across each column and computes
    the CLR transformation.

    :param matrix: A 2D NumPy array where each row is a sample and each column a feature.
    :return: A 2D NumPy array with the CLR-transformed values.
    """
    non_zero_min = np.min(matrix[matrix > 0], axis=0)
    matrix[matrix == 0] = non_zero_min  # Avoid division by zero
    gm = gmean(matrix, axis=0)
    clr_matrix = np.log(matrix / gm)
    return clr_matrix


def fit_kmedoids(
    data: np.ndarray,
    n_clusters: int = 5,
    random_state: int = 0,
    init: str = "k-medoids++",
    metric: str = "chebyshev",
) -> object:
    """Fit the K-Medoids clustering algorithm to the given data.

    This function applies the K-Medoids clustering algorithm using the given initialization method,
    number of clusters, metric, and random state.

    :param data: A 2D NumPy array where each row is a sample and each column a feature.
    :param n_clusters: The number of clusters to form (default is 5).
    :param random_state: The random seed used for initialization (default is 0).
    :param init: Initialization method for K-Medoids (default is 'k-medoids++').
    :param metric: The metric used to compute distances between points (default is 'chebyshev').
    :return: A fitted KMedoids object.
    """
    from sklearn_extra.cluster import KMedoids

    k_medoids = KMedoids(
        n_clusters=n_clusters, random_state=random_state, init=init, metric=metric
    ).fit(data)
    return k_medoids


def find_min_mean_indices(
    data: np.ndarray,
    k_medoids: object,
    n_clusters: int = 5,
    n_smallest_clusters: int = 1,
) -> list[np.ndarray]:
    """Find the indices of the clusters with the smallest mean values for each feature.

    This function calculates the mean value of each feature within each cluster. It then returns the
    indices of the clusters with the smallest means for each feature.

    :param data: A 2D NumPy array where each row is a sample and each column a feature.
    :param k_medoids: A fitted KMedoids object with the cluster labels.
    :param n_clusters: The total number of clusters (default is 5).
    :param n_smallest_clusters: The number of smallest-mean clusters to return per feature (default
        is 1).
    :return: A list of arrays, each containing the indices of the clusters with the smallest means
        for a feature.
    """
    return [
        np.argsort(
            [data[(k_medoids.labels_ == i), ii].mean() for i in range(n_clusters)]
        )[:n_smallest_clusters]
        for ii in range(data.shape[1])
    ]


def negbinomial_fit(x: np.ndarray) -> tuple[float, float, bool]:
    """Fit a Negative Binomial distribution to the given data.

    This function fits a Negative Binomial model to the input data using
    statsmodels. It returns the fitted mean, dispersion parameter, and a
    boolean indicating whether the model converged.

    :param x: A 1D NumPy array representing the data.
    :return: A tuple containing:
        - mu: The mean of the fitted distribution.
        - alpha: The dispersion parameter.
        - converged: A boolean indicating whether the fit converged.
    """
    res = sm.NegativeBinomial(x, np.ones_like(x)).fit(disp=0)
    if not res.converged:
        logger.warning("Negative Binomial fit did not converge")
    intercept, alpha = res.params
    mu = np.exp(intercept)
    return mu, alpha, res.converged


def compute_quantiles(
    pixel_array: np.ndarray,
    k_medoids: object,
    index_lst: list[np.ndarray],
    percentile_threshold: float = 99.5,
    quantile_val: float = 0.99,
) -> list[float]:
    """Compute quantiles for pixel intensities within selected clusters.

    This function selects data points based on k-medoids labels for each cluster, removes extreme
    outliers beyond a given percentile, fits a negative binomial model to the remaining data, and
    computes the desired quantile.

    :param pixel_array: A 2D array of pixel values with shape (n_samples, n_features).
    :param k_medoids: A fitted KMedoids object with cluster labels.
    :param index_lst: A list of arrays, each containing the indices of the clusters to analyze.
    :param percentile_threshold: The percentile threshold to filter out extreme values (default:
        99.5).
    :param quantile_val: The quantile value to compute from the negative binomial fit (default:
        0.99).
    :return: A list of quantiles for each feature.
    """
    lst = []
    for i in range(len(index_lst)):
        data = np.concatenate(
            [pixel_array[(k_medoids.labels_ == idx), i] for idx in index_lst[i]],
            axis=None,
        )
        threshold = np.percentile(data, percentile_threshold)
        data_preprocessed = data[data < threshold]

        if data_preprocessed.any():
            mu, alpha, converged = negbinomial_fit(data_preprocessed)
        else:
            converged = False

        if converged:
            quantile = scipy.stats.nbinom.ppf(
                quantile_val, 1 / alpha, 1 / (1 + mu * alpha)
            )
        else:
            quantile = np.nan
        lst.append(quantile)

    return lst


def compute_threshold(
    pixel_array: np.ndarray, n_clusters: int = 5, n_smallest_clusters: int = 1
) -> list[float]:
    """Compute thresholds for pixel intensities using k-medoids clustering.

    This function applies centered log-ratio (CLR) transformation to the pixel data, fits k-medoids
    clustering, identifies clusters with the smallest means, and computes quantiles for these
    clusters.

    :param pixel_array: A 2D array of pixel values with shape (n_samples, n_features).
    :param n_clusters: The number of clusters to form (default: 5).
    :param n_smallest_clusters: The number of smallest-mean clusters to consider per feature
        (default: 1).
    :return: A list of thresholds for each feature.
    """
    norm_px = clr_2d(pixel_array)
    k_medoids = fit_kmedoids(norm_px, n_clusters=n_clusters)
    index_lst = find_min_mean_indices(
        pixel_array,
        k_medoids,
        n_clusters=n_clusters,
        n_smallest_clusters=n_smallest_clusters,
    )
    thresholds = compute_quantiles(
        pixel_array, k_medoids, index_lst, percentile_threshold=99.5
    )
    return thresholds


def extract_peaks(
    img_data: np.ndarray, lbs_data: np.ndarray, dist_peaks: int
) -> tuple[np.ndarray, np.ndarray]:
    """Extract peaks from image data based on labeled regions.

    This function sets the intensity to zero where labels are zero, finds peaks
    for each channel, and returns the pixel values and corresponding cell labels
    for the detected peaks.

    :param img_data: A 3D array of image intensities with shape (channels, height, width).
    :param lbs_data: A 2D array of labeled regions with shape (height, width).
    :param dist_peaks: The minimum distance between peaks.
    :return: A tuple containing:
        - pixel_array: A 2D array of pixel values for detected peaks.
        - cell_labels: A 1D array of cell labels for the detected peaks.
    """
    img_data[..., lbs_data == 0] = 0
    peaks = np.sum([_find_peaks(i, n=dist_peaks) for i in img_data], axis=0)
    pixel_array = img_data[:, peaks > 0].T
    cell_labels = lbs_data[peaks > 0]
    return pixel_array, cell_labels


def process_spots(
    images_list: list[np.ndarray],
    labels_list: list[np.ndarray],
    thresholds: list[float],
    remove_all: bool = True,
) -> list[np.ndarray]:
    """Process and filter spots based on intensity thresholds across multiple channels.

    This function applies intensity thresholds to identify spots in each image channel.
    If `remove_all` is enabled, spots that are positive in all channels are removed.
    The function is not recommended for single-channel data.

    :param images_list: List of 3D arrays representing images, with shape (channels, height, width).
    :param labels_list: List of 2D arrays representing labeled regions, with shape (height, width).
    :param thresholds: List of intensity thresholds for each channel.
    :param remove_all: If True, remove spots positive in all channels (default: True).
    :return: A list of 2D arrays with filtered spot labels for each input image.

    :example:

    .. code-block:: python

        images = [np.random.rand(3, 100, 100) for _ in range(2)]
        labels = [np.random.randint(0, 2, (100, 100)) for _ in range(2)]
        thresholds = [0.5, 0.6, 0.7]

        spots = process_spots(images, labels, thresholds, remove_all=True)
        print(spots)
    """
    spots_list = []

    for img, lbs in zip(images_list, labels_list):
        # Identify spots exceeding the threshold in any channel
        arr = np.any([img[i] > thresh for i, thresh in enumerate(thresholds)], axis=0)

        if remove_all:
            # Remove spots that are positive in all channels
            arr = np.logical_and(
                arr,
                ~np.all(
                    [img[i] > thresh for i, thresh in enumerate(thresholds)], axis=0
                ),
            )

        # Apply labels where conditions are met
        spots = np.where(arr, lbs, 0)
        spots_list.append(spots)

    return spots_list
