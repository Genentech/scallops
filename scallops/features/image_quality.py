import numpy as np
import scipy
from scipy.fftpack import fft2
from scipy.ndimage import sum as nd_sum


# copied from centrosome.radial_power_spectrum but use np.ptp instead of img.ptp for numpy 2
def rps(img):
    assert img.ndim == 2
    radii2 = (np.arange(img.shape[0]).reshape((img.shape[0], 1)) ** 2) + (
        np.arange(img.shape[1]) ** 2
    )
    radii2 = np.minimum(radii2, np.flipud(radii2))
    radii2 = np.minimum(radii2, np.fliplr(radii2))
    maxwidth = (
        min(img.shape[0], img.shape[1]) / 8.0
    )  # truncate early to avoid edge effects
    if np.ptp(img) > 0:
        img = img / np.median(abs(img - img.mean()))  # intensity invariant
    mag = abs(fft2(img - np.mean(img)))
    power = mag**2
    radii = np.floor(np.sqrt(radii2)).astype(int) + 1
    labels = np.arange(2, np.floor(maxwidth)).astype(int).tolist()  # skip DC component
    if len(labels) > 0:
        magsum = nd_sum(mag, radii, labels)
        powersum = nd_sum(power, radii, labels)
        return np.array(labels), np.array(magsum), np.array(powersum)

    return [2], [0], [0]


def power_spectrum(image: np.ndarray) -> float:
    """The slope of the image log-log power spectrum.
    The power spectrum contains the frequency information of the image, and the slope
    gives a measure of image blur. A higher slope indicates more lower frequency
    components, and hence more blur. See https://cellprofiler-manual.s3.amazonaws.com/CellProfiler-4.0.5/modules/measurement.html#measureimagequality
    """
    radii, magnitude, power = rps(image)
    if sum(magnitude) > 0 and len(np.unique(image)) > 1:
        valid = magnitude > 0
        radii = radii[valid].reshape((-1, 1))
        power = power[valid].reshape((-1, 1))
        if radii.shape[0] > 1:
            idx = np.isfinite(np.log(power))
            return scipy.linalg.lstsq(
                np.hstack(
                    (
                        np.log(radii)[idx][:, np.newaxis],
                        np.ones(radii.shape)[idx][:, np.newaxis],
                    )
                ),
                np.log(power)[idx][:, np.newaxis],
            )[0][0][0]

    return 0.0
