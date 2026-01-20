import itertools
import logging
import math
from collections.abc import Callable, Sequence

import dask
import igraph as ig
import numpy as np
from dask import delayed
from numba import njit
from scipy import ndimage as ndi
from scipy.fft import fftfreq, fftn, ifftn
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger("scallops")
# Functions to calculate zero-normalized cross correlation and normalized cross correlation


@njit
def _collect_stats(ref, mov):
    mean_x = mean_y = sum_xy = sum_x2 = sum_y2 = 0.0
    for i in range(ref.shape[0]):
        for j in range(ref.shape[1]):
            x = float(ref[i][j])
            y = float(mov[i][j])
            mean_x += x
            mean_y += y
            sum_xy += x * y
            sum_x2 += x * x
            sum_y2 += y * y
    return mean_x, mean_y, sum_xy, sum_x2, sum_y2


@njit
def _calc_zncc(mean_x, mean_y, sum_xy, sum_x2, sum_y2, s_pixels) -> float:
    if s_pixels < 2.0:
        return 0.0
    mean_x /= s_pixels
    mean_y /= s_pixels
    denom = ((sum_x2 / s_pixels - mean_x**2) ** 0.5) * (
        (sum_y2 / s_pixels - mean_y**2) ** 0.5
    )
    zncc_value = (sum_xy / s_pixels - mean_x * mean_y) / denom if denom > 1e-15 else 0.0
    return zncc_value


@njit
def _zncc(ref: np.ndarray, mov: np.ndarray) -> float:
    mean_x, mean_y, sum_xy, sum_x2, sum_y2 = _collect_stats(ref, mov)
    return _calc_zncc(mean_x, mean_y, sum_xy, sum_x2, sum_y2, float(ref.size))


@njit
def _ncc(ref: np.ndarray, mov: np.ndarray) -> float:
    mean_x, mean_y, sum_xy, sum_x2, sum_y2 = _collect_stats(ref, mov)
    denom = (sum_x2**0.5) * (sum_y2**0.5)
    ncc_value = sum_xy / denom if denom > 1e-15 else 0.0
    return ncc_value


def ncc_after_shift(
    ref: np.ndarray, mov: np.ndarray, shift, order=1, ncc_func=_zncc
) -> float:
    mov_shift = ndi.shift(mov, shift=shift, order=order)

    pad_y = math.ceil(abs(shift[0]))
    pad_x = math.ceil(abs(shift[1]))
    slice_y = slice(pad_y, None) if shift[0] >= 0.0 else slice(None, -pad_y)
    slice_x = slice(pad_x, None) if shift[1] >= 0.0 else slice(None, -pad_x)

    return ncc_func(ref[slice_y, slice_x], mov_shift[slice_y, slice_x])


# Find shift trys to find the best shift to consider both phase and non-phase correlations


@njit
def _disam_shift(ref, mov, pos_shift, ncc_func=_zncc, frac_min=0.01):
    ## 4 cases, assume pos_shift is a np.array; note that ref does not need to be a square
    ## frac_min = 0.01, at least 1% of the area
    sy = ref.shape[0]
    sx = ref.shape[1]
    dy = pos_shift[0]
    dx = pos_shift[1]
    # A
    area_frac = (float(sy - dy) / sy) * (float(sx - dx) / sx)
    shift = pos_shift.copy()  # Default shift is case A
    max_corr = 0.0
    if area_frac >= frac_min:
        ref_crop = ref[dy:, dx:]
        mov_crop = mov[: sy - dy, : sx - dx]
        max_corr = ncc_func(ref_crop, mov_crop)
    # B
    area_frac = (float(sy - dy) / sy) * (float(dx) / sx)
    corr = 0.0
    if area_frac >= frac_min:
        ref_crop = ref[dy:, :dx]
        mov_crop = mov[: sy - dy, sx - dx :]
        corr = ncc_func(ref_crop, mov_crop)
    if corr > max_corr:
        shift[0] = dy
        shift[1] = dx - sx
        max_corr = corr
    # C
    area_frac = (float(dy) / sy) * (float(dx) / sx)
    corr = 0.0
    if area_frac >= frac_min:
        ref_crop = ref[:dy, :dx]
        mov_crop = mov[sy - dy :, sx - dx :]
        corr = ncc_func(ref_crop, mov_crop)
    if corr > max_corr:
        shift[0] = dy - sy
        shift[1] = dx - sx
        max_corr = corr
    # D
    area_frac = (float(dy) / sy) * (float(sx - dx) / sx)
    if area_frac >= frac_min:
        ref_crop = ref[:dy, dx:]
        mov_crop = mov[sy - dy :, : sx - dx]
    corr = ncc_func(ref_crop, mov_crop)
    if corr > max_corr:
        shift[0] = dy - sy
        shift[1] = dx
        max_corr = corr

    return shift, max_corr


def _upsampled_dft(data, upsampled_region_size, upsample_factor=1, axis_offsets=None):
    """Original skimage code
    Upsampled DFT by matrix multiplication.

    This code is intended to provide the same result as if the following
    operations were performed:
        - Embed the array "data" in an array that is ``upsample_factor`` times
          larger in each dimension.  ifftshift to bring the center of the
          image to (1,1).
        - Take the FFT of the larger array.
        - Extract an ``[upsampled_region_size]`` region of the result, starting
          with the ``[axis_offsets+1]`` element.

    It achieves this result by computing the DFT in the output array without
    the need to zeropad. Much faster and memory efficient than the zero-padded
    FFT approach if ``upsampled_region_size`` is much smaller than
    ``data.size * upsample_factor``.

    Parameters
    ----------
    data : array
        The input data array (DFT of original data) to upsample.
    upsampled_region_size : integer or tuple of integers, optional
        The size of the region to be sampled.  If one integer is provided, it
        is duplicated up to the dimensionality of ``data``.
    upsample_factor : integer, optional
        The upsampling factor.  Defaults to 1.
    axis_offsets : tuple of integers, optional
        The offsets of the region to be sampled.  Defaults to None (uses
        image center)

    Returns
    -------
    output : ndarray
            The upsampled DFT of the specified region.
    """
    # if people pass in an integer, expand it to a list of equal-sized sections
    if not hasattr(upsampled_region_size, "__iter__"):
        upsampled_region_size = [
            upsampled_region_size,
        ] * data.ndim
    else:
        if len(upsampled_region_size) != data.ndim:
            raise ValueError(
                "shape of upsampled region sizes must be equal "
                "to input data's number of dimensions."
            )

    if axis_offsets is None:
        axis_offsets = [
            0,
        ] * data.ndim
    else:
        if len(axis_offsets) != data.ndim:
            raise ValueError(
                "number of axis offsets must be equal to input "
                "data's number of dimensions."
            )

    im2pi = 1j * 2 * np.pi

    dim_properties = list(zip(data.shape, upsampled_region_size, axis_offsets))

    for n_items, ups_size, ax_offset in dim_properties[::-1]:
        kernel = (np.arange(ups_size) - ax_offset)[:, None] * fftfreq(
            n_items, upsample_factor
        )
        kernel = np.exp(-im2pi * kernel)
        # use kernel with same precision as the data
        kernel = kernel.astype(data.dtype, copy=False)

        # Equivalent to:
        #   data[i, j, k] = kernel[i, :] @ data[j, k].T
        data = np.tensordot(kernel, data, axes=(1, -1))
    return data


def _find_shift_no_upsample(ref, mov, ncc_func=_zncc, frac_min=0.01, center_img=False):
    # normalization == None
    src_freq = fftn(ref if not center_img else ref - ref.mean())
    target_freq = fftn(mov if not center_img else mov - mov.mean())
    image_product = src_freq * target_freq.conj()
    cc = ifftn(image_product)
    pos_shift = np.array(np.unravel_index(np.argmax(np.abs(cc)), cc.shape))
    shift, ncc_value = _disam_shift(
        ref, mov, pos_shift, ncc_func=ncc_func, frac_min=frac_min
    )

    # normalization == 'phase'
    eps = np.finfo(image_product.real.dtype).eps
    image_product2 = image_product / np.maximum(np.abs(image_product), 100 * eps)
    cc = ifftn(image_product2)
    pos_shift = np.array(np.unravel_index(np.argmax(np.abs(cc)), cc.shape))
    shift2, ncc2 = _disam_shift(
        ref, mov, pos_shift, ncc_func=ncc_func, frac_min=frac_min
    )

    # Find the best
    if ncc2 > ncc_value:
        shift = shift2
        ncc_value = ncc2
        image_product = image_product2

    return shift, ncc_value, image_product


def _upsample(image_product, shift, upsample_factor):
    # Initial shift estimate in upsampled grid
    upsample_factor = float(upsample_factor)
    shift = np.round(shift * upsample_factor) / upsample_factor
    upsampled_region_size = np.ceil(upsample_factor * 1.5)
    # Center of output array at dftshift + 1
    dftshift = np.fix(upsampled_region_size / 2.0)
    # Matrix multiply DFT around the current shift estimate
    sample_region_offset = dftshift - shift * upsample_factor
    cross_correlation = _upsampled_dft(
        image_product.conj(),
        upsampled_region_size,
        upsample_factor,
        sample_region_offset,
    ).conj()
    # Locate maximum and map back to original pixel grid
    maxima = np.array(
        np.unravel_index(np.argmax(np.abs(cross_correlation)), cross_correlation.shape),
        dtype=float,
    )
    maxima -= dftshift

    shift += maxima / upsample_factor

    return shift


def find_shift(
    ref, mov, upsample_factor=1, ncc_func=_zncc, frac_min=0.01, center_img=False
):
    # This function is adapted from skimage.registration._phase_cross_correlation.py
    # Try both phase and non-phase
    # frac_min is the minimal area fraction used for disambigute

    shift, ncc_value, image_product = _find_shift_no_upsample(
        ref, mov, ncc_func=ncc_func, frac_min=frac_min, center_img=center_img
    )

    # Upsampling
    if upsample_factor > 1:
        shift = _upsample(image_product, shift, upsample_factor)
        ncc_value = ncc_after_shift(ref, mov, shift, ncc_func=ncc_func)

    return shift, ncc_value


# Calculate best shift between two square tiles


@njit
def _get_crops(ref, mov, dy, dx, sy, sx):
    # Assume shift = (dy, dx) is integer here
    if dy >= 0:
        if dx >= 0:
            ref_crop = ref[dy:, dx:]
            mov_crop = mov[: sy - dy, : sx - dx]
        else:
            ref_crop = ref[dy:, : sx + dx]
            mov_crop = mov[: sy - dy, -dx:]
    else:
        if dx < 0:
            ref_crop = ref[: sy + dy, : sx + dx]
            mov_crop = mov[-dy:, -dx:]
        else:
            ref_crop = ref[: sy + dy, dx:]
            mov_crop = mov[-dy:, : sx - dx]

    return ref_crop, mov_crop


@njit
def _get_overlap_frac(dy, dx, sy, sx):
    return ((sy - dy if dy >= 0.0 else sy + dy) / float(sy)) * (
        (sx - dx if dx >= 0.0 else sx + dx) / float(sx)
    )


def calc_best_shift(
    ref: np.ndarray,
    mov: np.ndarray,
    orig_shift: tuple[float, float] = (0.0, 0.0),
    upsample_factor: int = 10,
    ncc_func: Callable[[np.ndarray, np.ndarray], float] = _zncc,
    overlap_min: float = 0.01,
    center_img: bool = False,
):
    # First find shift based on the crop of orig_shift and then do an upsample version using the updated shift crops
    # orig_shift = (orig_y, orig_x) = (mov.y-ref.y, mov.x-ref.x)
    # require the minimal overlapping be overlap_min
    assert ref.shape == mov.shape, f"{ref.shape} != {mov.shape}"
    sy = ref.shape[0]
    sx = ref.shape[1]

    ncc_value = 0.0
    final_shift = np.zeros(2, dtype=float)

    dy = int(orig_shift[0])
    dx = int(orig_shift[1])
    overlap_frac = _get_overlap_frac(dy, dx, sy, sx)

    if overlap_frac < overlap_min:
        # if the orig_shift leads to overlap < overlap_min, return with zncc = 0.0
        return final_shift, 0.0

    ref_crop, mov_crop = _get_crops(ref, mov, dy, dx, sy, sx)
    frac_min = (
        overlap_min / overlap_frac
    )  # consider relative area of the crop with respect to the tile
    shift, ncc1, image_product = _find_shift_no_upsample(
        ref_crop, mov_crop, ncc_func=ncc_func, frac_min=frac_min, center_img=center_img
    )
    crop_frac = (
        _get_overlap_frac(shift[0], shift[1], ref_crop.shape[0], ref_crop.shape[1])
        * overlap_frac
    )

    if ncc1 > 0.0 and crop_frac >= overlap_min:
        # If first attempt makes sense, try the second attempt
        dy2 = dy + round(shift[0])
        dx2 = dx + round(shift[1])
        overlap_frac2 = _get_overlap_frac(dy2, dx2, sy, sx)

        if overlap_frac2 < overlap_min:
            return final_shift, 0.0
        ref_crop2, mov_crop2 = _get_crops(ref, mov, dy2, dx2, sy, sx)
        ncc_value = ncc_func(ref_crop2, mov_crop2)  # NCC value after first attempt
        # Second attempt, if we can find better shift with global ncc?
        frac_min2 = overlap_min / overlap_frac2
        shift2, ncc2, image_product2 = _find_shift_no_upsample(
            ref_crop2,
            mov_crop2,
            ncc_func=ncc_func,
            frac_min=frac_min2,
            center_img=center_img,
        )
        crop_frac2 = (
            _get_overlap_frac(
                shift2[0], shift2[1], ref_crop2.shape[0], ref_crop2.shape[1]
            )
            * overlap_frac2
        )

        if ncc2 > 0.0 and crop_frac2 >= overlap_min:
            dy3 = dy2 + round(shift2[0])
            dx3 = dx2 + round(shift2[1])
            ref_crop3, mov_crop3 = _get_crops(ref, mov, dy3, dx3, sy, sx)
            ncc_value2 = ncc_func(ref_crop3, mov_crop3)

            if ncc_value < ncc_value2:
                ncc_value = ncc_value2
                dy = dy2
                dx = dx2
                shift = shift2
                image_product = image_product2

        if upsample_factor > 1.0:
            shift = _upsample(image_product, shift, upsample_factor)

        final_shift[0] = dy + shift[0]
        final_shift[1] = dx + shift[1]

        if upsample_factor > 1.0:
            ncc_value = ncc_after_shift(ref, mov, final_shift, ncc_func=ncc_func)

    return final_shift, ncc_value


# Deal with staged positions


def _detect_dir(ref, mov, staged_raw):
    # 0: not determined; 1: top; 2: bottom; 3: left; 4: right
    diff = staged_raw[mov] - staged_raw[ref]
    axis = np.argmax(np.abs(diff))
    if abs(diff[axis]) == 0.0 or abs(diff[1 - axis]) * 10.0 > abs(
        diff[axis]
    ):  # the long side should be at least 10 times larger than the short side
        return 0
    return 1 + (diff[axis] > 0.0) + 2 * axis


def _find_quintet(anchor, knn, staged_raw):
    if len(staged_raw) <= 4:
        return None
    nbrs = knn.kneighbors(
        staged_raw[anchor].reshape(1, 2),
        n_neighbors=5,
        return_distance=False,
    ).ravel()
    quintet = np.full(5, -1, dtype=int)
    quintet[0] = nbrs[0]  # this should be the anchor point
    for i in range(1, 5):
        dir_ = _detect_dir(anchor, nbrs[i], staged_raw)
        if quintet[dir_] >= 0:
            return None  # two with same direction, failed
        quintet[dir_] = nbrs[i]
    return quintet


def _check_quintet(quintet, seen):
    for tile in quintet:
        if tile in seen:
            return False
    return True


def _add_quintet(quintet, seen):
    for tile in quintet:
        seen.add(tile)


def _locate_anchor_quintets(staged_raw):
    knn = NearestNeighbors(algorithm="kd_tree")
    knn.fit(staged_raw)

    centroid = staged_raw.mean(axis=0).reshape(1, 2)
    anchor_candidates = knn.kneighbors(
        centroid, n_neighbors=min(100, len(staged_raw) - 1), return_distance=False
    ).ravel()  # find 100 nearest points

    seen = set()
    quintets = []

    for anchor in anchor_candidates:
        if anchor not in seen:
            quintet = _find_quintet(anchor, knn, staged_raw)
            if quintet is not None and _check_quintet(quintet, seen):
                quintets.append(quintet)
                _add_quintet(quintet, seen)
                if len(quintets) >= 5:
                    break

    return quintets


def _detect_flip(imgs, coords, min_ncc, max_shift):
    # imgs: images of the quintet
    # coords: converted coordinates in pixels
    # min_ncc: minimum zncc value
    # max_shift: maximum shift in pixels
    n_dir = 4
    template = [(1, 0, 0), (0, 2, 0), (3, 0, 1), (0, 4, 1)]

    _calc_best_shift_delayed = delayed(calc_best_shift)

    results = []
    origs = np.zeros((n_dir, 2), dtype=float)
    for i in range(n_dir):
        orig_shift = np.zeros(2, dtype=float)
        ref_id, mov_id, axis = template[i]
        origs[i] = coords[mov_id] - coords[ref_id]
        orig_shift[axis] = origs[i, axis]
        results.append(_calc_best_shift_delayed(imgs[ref_id], imgs[mov_id], orig_shift))
        results.append(_calc_best_shift_delayed(imgs[mov_id], imgs[ref_id], orig_shift))

    results = dask.compute(*results)

    def _filter_result(result, axis, orig, min_ncc, max_shift):
        shift, zncc = result
        if (
            zncc < min_ncc
            or abs(shift[axis] - orig[axis]) > max_shift[axis]
            or min(
                abs(shift[1 - axis] - orig[1 - axis]),
                abs(shift[1 - axis] + orig[1 - axis]),
            )
            > max_shift[1 - axis]
        ):
            return -1.0
        return zncc

    flip = np.zeros(n_dir, dtype=float)
    area_fracs = np.full(4, np.nan, dtype=float)
    img_shape = imgs[0].shape
    for i in range(n_dir):
        val1 = _filter_result(
            results[i * 2], template[i][2], origs[i], min_ncc, max_shift
        )
        val2 = _filter_result(
            results[i * 2 + 1], template[i][2], origs[i], min_ncc, max_shift
        )
        if val1 > 0.0 and val2 < 0.0:
            flip[i] = 1.0
            shift = results[i * 2][0]
            area_fracs[i] = _get_overlap_frac(
                shift[0], shift[1], img_shape[0], img_shape[1]
            )
        elif val1 < 0.0 and val2 > 0.0:
            flip[i] = -1.0
            shift = results[i * 2 + 1][0]
            area_fracs[i] = _get_overlap_frac(
                shift[0], shift[1], img_shape[0], img_shape[1]
            )

    flip_y = flip_x = 0.0
    if flip[0] * flip[1] >= 0.0:
        flip_y = flip[0] if flip[0] != 0.0 else flip[1]
    if flip[2] * flip[3] >= 0.0:
        flip_x = flip[2] if flip[2] != 0.0 else flip[3]

    area_frac = np.nan
    if np.isnan(area_fracs).sum() < area_fracs.size:
        area_frac = np.nanmedian(area_fracs)

    return flip_y, flip_x, area_frac


def _swap(imgs, coords, mshift):
    imgs_new = []
    coords_new = coords.copy()
    for i, j in enumerate([0, 3, 4, 1, 2]):
        imgs_new.append(imgs[j])
        coords_new[i, 0] = coords[j, 1]
        coords_new[i, 1] = coords[j, 0]
    mshift_new = np.array([mshift[1], mshift[0]])
    return imgs_new, coords_new, mshift_new


def determine_layout(
    read_images: Sequence,
    stage_positions: np.ndarray,
    image_spacing: tuple[float, float],
    max_shifts: Sequence[float] = (100.0,),  # max shift guard
    min_ncc: float = 0.55,  # adjust it to 0.55 to be safe
):
    anchor_quintets = _locate_anchor_quintets(stage_positions)
    for max_shift in max_shifts:
        mshift = np.array([max_shift, max_shift]) / image_spacing
        tile_shape = None
        for quintet in anchor_quintets:
            imgs = []
            for i in range(len(quintet)):
                imgs.append(read_images[quintet[i]])
            imgs = dask.compute(*imgs)
            tile_shape = imgs[0].shape
            coords = stage_positions[quintet] / image_spacing
            flip_y, flip_x, area_frac = _detect_flip(imgs, coords, min_ncc, mshift)
            if flip_y * flip_x != 0.0:
                return (
                    False,
                    flip_y,
                    flip_x,
                    area_frac,
                    tile_shape,
                    quintet[0],
                    max_shift,
                )  # in addition return one center tile
            imgs, coords, mshift = _swap(imgs, coords, mshift)  # swap y & x
            flip_y, flip_x, area_frac = _detect_flip(imgs, coords, min_ncc, mshift)
            if flip_y * flip_x != 0.0:
                return (
                    True,
                    flip_y,
                    flip_x,
                    area_frac,
                    tile_shape,
                    quintet[0],
                    max_shift,
                )
    logger.error("Unable to automatically determine stage layout.")
    if tile_shape is None:
        tile_shape = read_images[0].compute().shape

    center_tile = (
        anchor_quintets[0][0] if len(anchor_quintets) > 0 else len(stage_positions) // 2
    )
    return False, 1, 1, 0.1, tile_shape, center_tile, max_shifts[-1]  # defaults


def convert_stage_positions(staged, swap, flip_y, flip_x, pixel_size):
    converted = staged[:, [1, 0]] if swap else staged.copy()
    converted[:, 0] *= flip_y
    converted[:, 1] *= flip_x
    converted /= pixel_size
    return converted


@njit
def _overlap_two_tiles(fov1, fov2, coord1, coord2, tile_shape):
    if coord1[0] > coord2[0]:
        fov1, fov2 = fov2, fov1
        coord1, coord2 = coord2, coord1
    shift = coord2 - coord1
    if shift[0] >= tile_shape[0]:
        return 0.0, None, None, None
    if shift[1] >= 0.0:
        if shift[1] >= tile_shape[1]:
            return 0.0, None, None, None
        return (
            ((tile_shape[0] - shift[0]) / tile_shape[0])
            * ((tile_shape[1] - shift[1]) / tile_shape[1]),
            fov1,
            fov2,
            shift,
        )
    else:
        if shift[1] <= -tile_shape[1]:
            return 0.0, None, None, None
        return (
            ((tile_shape[0] - shift[0]) / tile_shape[0])
            * ((tile_shape[1] + shift[1]) / tile_shape[1]),
            fov1,
            fov2,
            shift,
        )


@njit
def _find_all_overlap(coords, tile_shape, frac_thre=0.0):
    """Find all overlaps between tiles"""
    # Only return overlaps with area > frac_thre
    n = coords.shape[0]
    fracs = []
    pairs = []
    shifts = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            frac_area, fov1, fov2, shift = _overlap_two_tiles(
                i, j, coords[i], coords[j], tile_shape
            )
            if frac_area > frac_thre:
                fracs.append(frac_area)
                pairs.append([fov1, fov2])
                shifts.append(shift)

    return fracs, pairs, shifts


def find_overlaps(coords, tile_shape, frac_thre=0.0):
    # Only return overlaps with area > frac_thre
    fracs, pairs, shifts = _find_all_overlap(coords, tile_shape, frac_thre=frac_thre)
    return np.array(fracs), np.array(pairs), np.array(shifts)


def sample_null(
    n_tiles: int,
    pairs: np.ndarray,
    shifts: np.ndarray,
    n_samples: int = 1000,
    seed: int = 239753,
):
    """Random sample non-overlaping tiles with real shifts to generate null distribution"""

    rng = np.random.default_rng(seed=seed)
    forbidden = set()
    n_pairs = len(pairs)

    # exclude actual overlapping pairs
    for i in range(n_pairs):
        forbidden.add((pairs[i, 0], pairs[i, 1]))
        forbidden.add((pairs[i, 1], pairs[i, 0]))

    if n_tiles < 50:
        # generate all unique combinations
        n_samples = math.comb(n_tiles, 2)
        if n_samples < 50:
            # include actual overlapping pairs when small number of samples
            forbidden.clear()

        null_pairs = np.zeros((n_samples, 2), dtype=int)
        null_shifts = np.zeros((n_samples, 2), dtype=float)
        combo_index = 0
        for x, y in itertools.combinations(np.arange(n_tiles), 2):
            if (x, y) not in forbidden:
                null_pairs[combo_index, 0] = x
                null_pairs[combo_index, 1] = y
                pos_shift = rng.integers(n_pairs)
                null_shifts[combo_index] = shifts[pos_shift]
                combo_index += 1
        if combo_index < n_samples:
            null_pairs = null_pairs[0:combo_index]
            null_shifts = null_shifts[0:combo_index]
    else:
        max_tries = 1000
        null_pairs = np.zeros((n_samples, 2), dtype=int)
        null_shifts = np.zeros((n_samples, 2), dtype=float)
        for i in range(n_samples):
            found = False
            for j in range(max_tries):
                x = rng.integers(n_tiles)
                y = rng.integers(n_tiles)
                if (x != y) and ((x, y) not in forbidden):
                    found = True
                    break
            if found:
                forbidden.add((x, y))
                null_pairs[i, 0] = x
                null_pairs[i, 1] = y
                pos_shift = rng.integers(n_pairs)
                null_shifts[i] = shifts[pos_shift]
            else:
                raise ValueError("Unable to compute null")

    return null_pairs, null_shifts


def calc_mu_sigma_null(null_values) -> tuple[float, float]:
    mu = np.median(null_values)
    abs_dev = np.abs(null_values - mu)
    mad = np.median(abs_dev)
    sigma = mad * 1.4826

    return (mu, sigma)


def _build_graph(n, edges, ncc_values, deltas):
    g = ig.Graph(directed=True)
    g.add_vertices(n, attributes={"orig_id": list(range(n))})
    g.add_edges(edges, attributes={"ncc": ncc_values, "delta": deltas})
    return g


def _get_center(g):
    ecc = np.array(g.eccentricity(mode="all"))  # ignore edge direction
    min_ecc = ecc.min()
    center_id = np.where(ecc == min_ecc)[0][0]
    return center_id


def _find_spanning_trees(g):
    cc = g.connected_components(mode="weak")  # ignore edge direction
    spanning_trees = []
    max_id = -1
    max_n = 0
    for i, component in enumerate(cc):
        n_v = len(component)
        gc = g.induced_subgraph(component)

        if n_v > 1:
            center = _get_center(gc)
            errors = -np.log(
                gc.es["ncc"]
            )  # negative log as errors, assume ncc <=0 are all filtered already
            paths = gc.get_all_shortest_paths(
                center, weights=errors, mode="all"
            )  # ignore edge direction

            eids = []
            for j in range(len(paths)):
                assert j == paths[j][-1]
                if len(paths[j]) > 1:
                    eids.append(
                        gc.get_eid(paths[j][-2], paths[j][-1], directed=False)
                    )  # ignore edge direction
            assert len(eids) == gc.vcount() - 1  # is a tree
            stree = gc.subgraph_edges(eids)  # note the graph is still directed

            spanning_trees.append((center, stree, gc))
        else:
            spanning_trees.append((0, gc, gc))

        if max_n < n_v:
            max_n = n_v
            max_id = i

    return spanning_trees, max_id


def _obj(x, deltas, edges, coefs):
    return (coefs * ((x[edges[:, 1]] - x[edges[:, 0]] - deltas) ** 2)).sum()


def _propogate_delta(n_tiles, spanning_trees, power_coef, maxiter):
    # power_coef: raise to x^power_coef when compute L2 norm
    # maxiter: max iteration for optimization
    adjusts = np.zeros((n_tiles, 2), dtype=float)

    for stree in spanning_trees:
        tree = stree[1]
        if tree.vcount() > 1:
            adjusts_c = np.zeros(
                (tree.vcount(), 2), dtype=float
            )  # local adjusts for this component

            # First to get adjusts using the spanning tree
            root = stree[0]
            vertices, layers, parents = tree.bfs(root, mode="all")
            for v in vertices:
                if parents[v] >= 0:
                    eid = tree.get_eid(parents[v], v, directed=False)
                    if tree.es[eid].target == v:
                        adjusts_c[v] = adjusts_c[parents[v]] + tree.es[eid]["delta"]
                    else:
                        adjusts_c[v] = adjusts_c[parents[v]] - tree.es[eid]["delta"]

            # Use L2 norm to consider all edges
            graph = stree[2]
            deltas = np.array(graph.es["delta"])
            edges = np.array(graph.get_edgelist())
            coefs = np.array(graph.es["ncc"]) ** power_coef

            y_optim = minimize(
                _obj,
                adjusts_c[:, 0],
                args=(deltas[:, 0], edges, coefs),
                options={"maxiter": maxiter},
            ).x
            x_optim = minimize(
                _obj,
                adjusts_c[:, 1],
                args=(deltas[:, 1], edges, coefs),
                options={"maxiter": maxiter},
            ).x

            y_optim -= y_optim[root]  # Make sure center has 0 adjustment
            x_optim -= x_optim[root]

            adjusts[graph.vs["orig_id"], 0] = y_optim
            adjusts[graph.vs["orig_id"], 1] = x_optim

    return adjusts


def _fit_model(spanning_trees, max_id, staged, stitched):
    ### Use the same method as Ashlar
    regressor = LinearRegression()
    points = spanning_trees[max_id][2].vs["orig_id"]
    regressor.fit(staged[points], stitched[points])
    if np.linalg.det(regressor.coef_) < 1e-3:
        logger.info("Unable to fit model.")
        return

    for i in range(len(spanning_trees)):
        if i != max_id:
            points = spanning_trees[i][2].vs["orig_id"]
            centroid_input = staged[points].mean(axis=0)
            shift = regressor.predict([centroid_input])[0] - stitched[points].mean(
                axis=0
            )
            stitched[points] += shift


def get_stitched_positions(
    n_tiles,
    staged,
    pairs,
    delta_shifts,
    ncc_values,
    null_params,
    alpha=0.001,
    set_origin=True,
    power_coef=10.0,
    maxiter=5,
    max_shift_pixels=None,
):
    # delta_shifts here is the difference with the staged position, not shift of mov w.r.t. ref
    # set_origin == True will set the top left corner of the stitched image as (0, 0)
    # power_coef: raise to x^power_coef when compute L2 norm
    # maxiter: max iteration for optimization
    stitched = staged.copy()
    z_threshold = norm.ppf(1.0 - alpha)
    valid_edges = ((ncc_values - null_params[0]) / null_params[1]) > z_threshold

    n_valid_edges = valid_edges.sum()
    if n_valid_edges == 0:
        logger.info("No valid edges found. Using stage positions.")
        if set_origin:
            origin = stitched.min(axis=0)
            stitched -= origin
        return stitched, None, z_threshold, valid_edges, None
    logger.info(
        f"Kept {n_valid_edges:,} / {len(ncc_values):,} edges after filtering using "
        f"null distribution."
    )
    if max_shift_pixels is not None:
        valid_edges = valid_edges & np.all(
            np.abs(delta_shifts) <= max_shift_pixels, axis=1
        )
        n_valid_edges_before_max_shift_filter = n_valid_edges
        n_valid_edges = valid_edges.sum()
        logger.info(
            f"Kept {n_valid_edges:,} / {n_valid_edges_before_max_shift_filter:,} edges "
            f"after filtering using max shift."
        )
        if n_valid_edges == 0:
            if set_origin:
                origin = stitched.min(axis=0)
                stitched -= origin
            return stitched, None, z_threshold, valid_edges, None

    g = _build_graph(
        n_tiles, pairs[valid_edges], ncc_values[valid_edges], delta_shifts[valid_edges]
    )
    spanning_trees, max_id = _find_spanning_trees(g)

    stree = spanning_trees[max_id][1]

    mst_original_ids = stree.vs["orig_id"]
    spanning_tree_edges = []
    for edge in stree.get_edgelist():
        spanning_tree_edges.append(
            (mst_original_ids[edge[0]], mst_original_ids[edge[1]])
        )

    logger.info(f"Edges in spanning tree: {stree.ecount():,} / {g.ecount():,}.")

    adjusts = _propogate_delta(
        n_tiles, spanning_trees, power_coef=power_coef, maxiter=maxiter
    )
    stitched += adjusts

    if len(spanning_trees) > 1:
        logger.info(
            f"Found {len(spanning_trees):,} disconnected components. Fitting tile "
            f"positions with regression model."
        )
        _fit_model(spanning_trees, max_id, staged, stitched)
    final_shift = stitched - staged
    if set_origin:
        origin = stitched.min(axis=0)
        stitched -= origin

    return stitched, spanning_tree_edges, z_threshold, valid_edges, final_shift


def _get_overlap_bounding_box(y0, x0, dy, dx, sy, sx):
    """Convert overlap to bounding box.

    :param y0, x0: coordinate of pairs[i, 0]
    :param dy, dx: shifts[i]
    :param sy, sx: shape of one tile
    """

    topleft = np.array([y0 + max(0, dy), x0 + max(0, dx)])
    bottomright = np.array([y0 + sy + min(0, dy), x0 + sx + min(0, dx)])
    return topleft, bottomright


def _estimate_crop_width(fracs, min_frac, tile_shape, crop_width):
    pct = max(
        np.median(fracs[fracs > min_frac]) / 2.0 - 0.015, 0.0
    )  # 0.015 can be adjusted
    return (0 if crop_width is None else crop_width) + int(tile_shape[0] * pct)
