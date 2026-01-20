import centrosome.radial_power_spectrum
import numpy as np
import scipy


def power_spectrum(image: np.ndarray) -> float:
    """The slope of the image log-log power spectrum.
    The power spectrum contains the frequency information of the image, and the slope
    gives a measure of image blur. A higher slope indicates more lower frequency
    components, and hence more blur. See https://cellprofiler-manual.s3.amazonaws.com/CellProfiler-4.0.5/modules/measurement.html#measureimagequality
    """
    radii, magnitude, power = centrosome.radial_power_spectrum.rps(image)
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
