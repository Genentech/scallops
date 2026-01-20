"""Radial correction from ashlar"""

import numpy as np
from skimage.transform import warp
from skimage.util import img_as_float


def _radial_mapping(xy, center, k):
    x, y = xy.T.astype(float)
    y0, x0 = center
    x -= x0
    y -= y0
    # The warp mapping function defines the INVERSE map to apply, which in the
    # case of a distortion correction is the FORWARD map of the distortion
    # itself. See Fitzgibbon 2001 eq 1 for details.
    r2 = x**2 + y**2
    f = 1 + k * r2
    xy[..., 0] = x * f + x0
    xy[..., 1] = y * f + y0
    return xy


def radial_correct(
    image: np.ndarray,
    k: float,
    cval: float = 0,
) -> np.ndarray:
    """Perform a transformation to correct for barrel or pincushion distortion.

     Radial distortion is modeled using a one-parameter model described in
     A. W. Fitzgibbon, "Simultaneous linear estimation of multiple view
        geometry and lens distortion," Proceedings of the 2001 IEEE Computer
        Society Conference on Computer Vision and Pattern Recognition. CVPR 2001,
        Kauai, HI, USA, 2001, pp. I-I. :DOI:`10.1109/CVPR.2001.990465`. (Equation 1).

     :param image: Input image.
     :param k: Distortion parameter
     :param cval: The value outside the image boundaries.

    :return: Radial distortion corrected image.
    """

    mode = "constant"
    center = np.array(image.shape)[-2:] / 2
    warp_args = {"center": center, "k": k}
    order = 1
    clip = True
    preserve_range = False
    if image.ndim == 3:
        image = img_as_float(image, force_copy=True)
        for c in range(image.shape[0]):
            image[c] = warp(
                image[c],
                _radial_mapping,
                map_args=warp_args,
                output_shape=None,
                order=order,
                mode=mode,
                cval=cval,
                clip=clip,
                preserve_range=preserve_range,
            )
        return image
    assert image.ndim == 2, (
        f"Image must be 2 or 3 dimensional, but image has {image.ndim} dimensions."
    )

    return warp(
        image,
        _radial_mapping,
        map_args=warp_args,
        output_shape=None,
        order=order,
        mode=mode,
        cval=cval,
        clip=clip,
        preserve_range=preserve_range,
    )
