"""SCALLOPS module to visualize registration and image processing.

This module provides utilities for color map manipulations and validations,
including generating random colors in different color spaces and validating RGB colors.

Authors:
    - The SCALLOPS development team
"""

import warnings

import numpy as np
import skimage.color as colorconv

# code from napari
# obtained with colorconv.rgb2luv(_all_rgb().reshape((-1, 256, 3)))
LUVMIN = np.array([0.0, -83.07790815, -134.09790293])
LUVMAX = np.array([100.0, 175.01447356, 107.39905336])
LUVRNG = LUVMAX - LUVMIN

# obtained with colorconv.rgb2lab(_all_rgb().reshape((-1, 256, 3)))
LABMIN = np.array([0.0, -86.18302974, -107.85730021])
LABMAX = np.array([100.0, 98.23305386, 94.47812228])
LABRNG = LABMAX - LABMIN


def _validate_rgb(colors, *, tolerance=0.0):
    """Return the subset of colors that is in [0, 1] for all channels.

    :param colors: array of float, shape (N, 3)
        Input colors in RGB space.
    :param tolerance: float, optional
        Values outside of the range by less than `tolerance` are allowed and
        clipped to be within the range.
    :return: array of float, shape (M, 3), M <= N
        The subset of colors that are in valid RGB space.

    :example:

    .. code-block:: python

        colors = np.array([[0.0, 1.0, 1.0], [1.1, 0.0, -0.03], [1.2, 1.0, 0.5]])
        _validate_rgb(colors)
        # array([[0., 1., 1.]])
    """
    lo = 0 - tolerance
    hi = 1 + tolerance
    valid = np.all((colors >= lo) & (colors <= hi), axis=1)
    filtered_colors = np.clip(colors[valid], 0, 1)
    return filtered_colors


def _color_random(n, *, colorspace="lab", tolerance=0.0, seed=0.5):
    """Generate n random RGB colors uniformly from LAB or LUV space.

    :param n: int
        Number of colors to generate.
    :param colorspace: str, one of {'lab', 'luv', 'rgb'}
        The colorspace from which to get random colors.
    :param tolerance: float
        How much margin to allow for out-of-range RGB values (these are
        clipped to be in-range).
    :param seed: float or array of float, shape (3,)
        Value from which to start the quasirandom sequence.
    :return: array of float, shape (n, 3)
        RGB colors chosen uniformly at random from given colorspace.

    :example:

    .. code-block:: python

        colors = _color_random(5, colorspace="lab")
        print(colors)
    """
    factor = 6  # about 1/5 of random LUV tuples are inside the space
    expand_factor = 2
    rgb = np.zeros((0, 3))
    while len(rgb) < n:
        random = _low_discrepancy(3, n * factor, seed=seed)
        if colorspace == "luv":
            raw_rgb = colorconv.luv2rgb(random * LUVRNG + LUVMIN)
        elif colorspace == "rgb":
            raw_rgb = random
        else:  # 'lab' by default
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw_rgb = colorconv.lab2rgb(random * LABRNG + LABMIN)
        rgb = _validate_rgb(raw_rgb, tolerance=tolerance)
        factor *= expand_factor
    return rgb[:n]


def label_colormap(num_colors=256, seed=0.5):
    """Produce a colormap suitable for use with a given label set.

    :param num_colors: int, optional
        Number of unique colors to use. Default used if not given.
        Colors are in addition to a transparent color 0.
    :param seed: float or array of float, length 3
        The seed for the random color generator.
    :return: list of colors

    :example:

    .. code-block:: python

        colormap = label_colormap(num_colors=10)
        print(colormap)
    """
    colors = np.concatenate(
        (
            _color_random(num_colors + 1, seed=seed),
            np.full((num_colors + 1, 1), 1),
        ),
        axis=1,
    )
    # Insert alpha at layer 0
    colors[0, :] = 0  # ensure alpha is 0 for label 0
    return colors


def _low_discrepancy(dim, n, seed=0.5):
    """Generate a 1d, 2d, or 3d low discrepancy sequence of coordinates.

    :param dim: one of {1, 2, 3}
        The dimensionality of the sequence.
    :param n: int
        How many points to generate.
    :param seed: float or array of float, shape (dim,)
        The seed from which to start the quasirandom sequence.
    :return: array of float, shape (n, dim)
        The sampled points.

    References
    ----------
    .. [1]: http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/

    :example:

    .. code-block:: python

        pts = _low_discrepancy(2, 100)
        print(pts)
    """
    phi1 = 1.6180339887498948482
    phi2 = 1.32471795724474602596
    phi3 = 1.22074408460575947536
    seed = np.broadcast_to(seed, (1, dim))
    phi = np.array([phi1, phi2, phi3])
    g = 1 / phi
    n = np.reshape(np.arange(n), (n, 1))
    pts = (seed + (n * g[:dim])) % 1
    return pts
