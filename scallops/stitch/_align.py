import logging
from collections.abc import Callable, Sequence
from typing import Literal

import dask
import numpy as np
from dask import delayed
from dask.diagnostics import ProgressBar

from scallops.stitch._radial import radial_correct
from scallops.stitch._radial_optim import (
    _radial_crop_width,
    _sample_random_pairs,
    parallel_find_radial_K,
)
from scallops.stitch.shift_utils import (
    _calc_zncc,
    _collect_stats,
    _estimate_crop_width,
    _get_crops,
    _zncc,
    calc_best_shift,
    calc_mu_sigma_null,
    convert_stage_positions,
    determine_layout,
    find_overlaps,
    get_stitched_positions,
    sample_null,
)
from scallops.stitch.utils import _read_tile_delayed

logger = logging.getLogger("scallops")


def _zncc_shift_pair(
    ref: np.ndarray,
    mov: np.ndarray,
    shift: np.ndarray,
    upsample_factor=1.0,
    ncc_func=_zncc,
    overlap_min=0.01,
):
    shift, nccv = calc_best_shift(
        ref,
        mov,
        shift,
        upsample_factor=upsample_factor,
        ncc_func=ncc_func,
        overlap_min=overlap_min,
    )
    return nccv, shift


def _all_zncc_shifts(
    pairs: np.ndarray,
    shifts: np.ndarray,
    read_images: Sequence,
    upsample_factor=1.0,
    ncc_func=_zncc,
    overlap_min=0.01,
) -> tuple[np.ndarray, np.ndarray]:
    _zncc_shift_pair_delayed = delayed(_zncc_shift_pair, nout=2)

    results = []

    for i in range(len(pairs)):
        ref = read_images[pairs[i][0]]
        mov = read_images[pairs[i][1]]
        results.append(
            _zncc_shift_pair_delayed(
                ref=ref,
                mov=mov,
                shift=shifts[i],
                upsample_factor=upsample_factor,
                ncc_func=ncc_func,
                overlap_min=overlap_min,
            )
        )
    with ProgressBar():
        results = dask.compute(*results)
    ncc_values = np.array([x[0] for x in results])
    shifts = np.array([x[1] for x in results])
    return ncc_values, shifts


def _eval_pair(
    ref: np.ndarray,
    mov: np.ndarray,
    shift: np.ndarray,
):
    sy, sx = ref.shape
    dy = round(shift[0])
    dx = round(shift[1])
    ref_crop, mov_crop = _get_crops(ref, mov, dy, dx, sy, sx)
    mean_x, mean_y, sum_xy, sum_x2, sum_y2 = _collect_stats(ref_crop, mov_crop)
    s_pixels = float(ref_crop.size)
    zncc_val = _calc_zncc(mean_x, mean_y, sum_xy, sum_x2, sum_y2, s_pixels)
    return np.array([zncc_val, mean_x, mean_y, sum_xy, sum_x2, sum_y2, s_pixels])


def _eval_all_shifts(
    pairs: np.ndarray,
    shifts: np.ndarray,
    read_images: Sequence,
) -> tuple[float, np.ndarray]:
    _eval_pair_delayed = delayed(_eval_pair)
    results = []
    for i in range(len(pairs)):
        ref = read_images[pairs[i][0]]
        mov = read_images[pairs[i][1]]
        results.append(_eval_pair_delayed(ref=ref, mov=mov, shift=shifts[i]))
    with ProgressBar():
        results = dask.compute(*results)
    results = np.array(results)
    mean_x, mean_y, sum_xy, sum_x2, sum_y2, s_pixels = results[:, 1:].sum(axis=0)
    zncc_val = _calc_zncc(mean_x, mean_y, sum_xy, sum_x2, sum_y2, s_pixels)
    return zncc_val, results[:, 0].copy()


def _get_read_images(
    filepaths: Sequence[Sequence[str]],
    fileattrs: Sequence[dict[str, str | list[str]]] | None,
    channel: int = 0,
    z_index: int | Literal["max"] | Sequence[int] = "max",
    radial_correction_k: float | None = None,
    crop_width: int | None = None,
    n_scenes: int | None = None,
):
    n = len(filepaths) if n_scenes is None else n_scenes
    z_index_is_sequence = isinstance(
        z_index, (Sequence, np.ndarray)
    ) and not isinstance(z_index, str)

    return [
        _read_tile_delayed(
            file_list=filepaths[0] if n_scenes is not None else filepaths[i],
            attrs=fileattrs[i] if fileattrs is not None else None,
            channel=channel,
            crop_width=crop_width,
            scene_id=i if n_scenes is not None else None,
            radial_correction_k=radial_correction_k,
            z_index=z_index[i] if z_index_is_sequence else z_index,
        )
        for i in range(n)
    ]


def stitch_align(
    filepaths: Sequence[Sequence[str]],
    fileattrs: Sequence[dict[str, str | list[str]]] | None,
    image_spacing: tuple[float, float],
    stage_positions: np.ndarray,
    channel: int = 0,
    z_index: int | Sequence[int] | Literal["max"] = "max",
    radial_correction_k: float | None | Literal["auto"] = "auto",
    crop_width: int | None = None,
    upsample_factor=1.0,
    min_overlap_fraction: float = None,
    ncc_func: Callable[[np.ndarray, np.ndarray], float] = _zncc,
    stitch_alpha: float = 0.001,
    evaluate_stitching: bool = True,
    n_scenes: int | None = None,
    random_seed: int = 239753,
    max_shifts: Sequence[float] = (50,),
    flip_y_axis: int | None = None,
    flip_x_axis: int | None = None,
    swap_axes: bool | None = None,
) -> dict:
    """Find optimal tile positions

    :param filepaths: Sequence of tile paths
    :param fileattrs: Sequence of file metadata
    :param radial_correction_k: Radial correction k
    :param crop_width: Image crop width
    :param channel: Image channel to use for stitching
    :param image_spacing: Image spacing
    :param stage_positions: Image stage positions in microns.
    :param z_index: z-index or 'max'
    :param upsample_factor: Upsample factor for cross correlation
    :param min_overlap_fraction: Min overlap fraction to consider overlaps
    :param ncc_func: NCC function
    :param stitch_alpha: Stitching alpha
    :param evaluate_stitching: Whether to evaluate stitching quality
    :param n_scenes: Number of scenes when only one image provided with multiple scenes.
    :param random_seed: Random seed
    :param max_shifts: Maximum allowed per tile shift in microns
    :param flip_y_axis: Whether to flip y axis and override automatic determination
    :param flip_x_axis: Whether to flip x axis and override automatic determination
    :param swap_axes: Whether to swap y and x axes and override automatic determination
    :return: Result dictionary
    """

    read_images = _get_read_images(
        filepaths=filepaths,
        fileattrs=fileattrs,
        channel=channel,
        z_index=z_index,
        radial_correction_k=None,  # Do not do radial correction before determine layout
        crop_width=crop_width,
        n_scenes=n_scenes,
    )

    (
        swap_axes_auto,
        flip_y_auto,
        flip_x_auto,
        area_fraction_auto,
        tile_shape_no_crop,
        center_tile,
        max_shift,
    ) = determine_layout(
        read_images=read_images,
        stage_positions=stage_positions,
        image_spacing=image_spacing,
        max_shifts=max_shifts,
    )
    if swap_axes is None:
        swap_axes = swap_axes_auto
    if flip_y_axis is None:
        flip_y_axis = flip_y_auto
    if flip_x_axis is None:
        flip_x_axis = flip_x_auto

    original_tile_shape = tile_shape_no_crop
    if crop_width is not None:
        original_tile_shape = (
            original_tile_shape[0] + crop_width * 2,
            original_tile_shape[1] + crop_width * 2,
        )

    if min_overlap_fraction is None:
        # Make a loose cutoff, alternatively, can use no threshold
        min_overlap_fraction = area_fraction_auto / 3.0
    logger.info(
        f"Flip y axis: {str(flip_y_axis == -1).lower()}, "
        f"flip x axis: {str(flip_x_axis == -1).lower()}, swap axes: {str(swap_axes).lower()}, "
        f"and minimum tile overlap: {min_overlap_fraction:.2f}."
    )
    # Convert the raw staged position into image coordinate system
    staged = convert_stage_positions(
        stage_positions, swap_axes, flip_y_axis, flip_x_axis, image_spacing
    )

    n_tiles = staged.shape[0]

    # Graph Building Preparation

    # find all overlaps between tiles with at least `min_overlap_fraction` of area
    # using stage positions
    fracs, pairs, orig_shifts = find_overlaps(
        staged, tile_shape_no_crop, min_overlap_fraction
    )
    tile_shape = tile_shape_no_crop
    max_shift_pixels = (
        np.array([max_shift, max_shift]) / image_spacing
        if max_shift is not None
        else None
    )
    logger.info(f"Found {len(pairs):,} edges.")
    if len(pairs) == 0:
        logger.info("No valid edges found. Using stage positions.")
        origin = staged.min(axis=0)
        staged -= origin
        return dict(
            swap=swap_axes,
            flip_y=flip_y_axis,
            flip_x=flip_x_axis,
            tile_shape=original_tile_shape,
            align_tile_shape=tile_shape,
            fuse_crop_width=0,
            max_shift=max_shift,
            crop_width=crop_width,
            min_overlap_fraction=min_overlap_fraction,
            area_fraction=area_fraction_auto,
            radial_correction_k=radial_correction_k,
            stitched_positions=staged,
            pairs=pairs,
            nccs=None,
            null_params=None,
            null_nccs=None,
            delta_shifts=None,
            zncc_val=None,
            zncc_values=None,
            pairs_after_stitching=None,
            shifts_after_stitching=None,
            fractions_after_stitching=None,
            spanning_tree_edges=None,
            final_shifts=None,
            z_threshold=None,
            valid_edges=None,
        )

    if radial_correction_k is not None:
        if radial_correction_k == "auto":
            logger.info("Computing radial correction K.")
            n_pairs = min(len(pairs), 21)
            if n_pairs % 2 == 0:
                n_pairs -= 1
            # Make sure n_pairs is odd
            radial_correction_k, crop_width_ = parallel_find_radial_K(
                _sample_random_pairs(pairs, seed=random_seed, size=n_pairs),
                read_images,
                pairs,
                orig_shifts,
                upsample_factor=upsample_factor,
            )
        else:
            crop_width_ = _radial_crop_width(
                radial_correct(read_images[center_tile].compute(), radial_correction_k)
            )

        if crop_width is None:
            crop_width = crop_width_

        # update read image to use barrel correction k and crop_width
        read_images = _get_read_images(
            filepaths=filepaths,
            fileattrs=fileattrs,
            channel=channel,
            z_index=z_index,
            radial_correction_k=radial_correction_k,
            crop_width=crop_width,
            n_scenes=n_scenes,
        )

        tile_shape = (
            read_images[0].compute().shape
        )  # Update tile_shape in case cropping happens
        logger.info(
            f"Radial correction K: {radial_correction_k}, crop width: {crop_width}."
        )

    #  Compute the null
    null_pairs, null_shifts = sample_null(n_tiles, pairs, orig_shifts, seed=random_seed)
    logger.info("Computing null distribution.")
    null_nccs, _ = _all_zncc_shifts(
        read_images=read_images,
        pairs=null_pairs,
        shifts=null_shifts,
        upsample_factor=upsample_factor,
        ncc_func=ncc_func,
    )

    null_params = calc_mu_sigma_null(null_nccs)
    logger.info(
        f"Null distribution median: {null_params[0]:.4f}, standard deviation estimated "
        f"from MAD: {null_params[1]:.4f}."
    )
    # Compute all pair-wise zncc
    logger.info("Computing tile shifts.")
    nccs, updated_shifts = _all_zncc_shifts(
        read_images=read_images,
        pairs=pairs,
        shifts=orig_shifts,
        upsample_factor=upsample_factor,
        ncc_func=ncc_func,
    )

    # Compute stitched positions
    logger.info("Optimizing tile positions.")
    delta_shifts = updated_shifts - orig_shifts

    stitched, spanning_tree_edges, z_threshold, valid_edges, final_shifts = (
        get_stitched_positions(
            n_tiles=n_tiles,
            staged=staged,
            pairs=pairs,
            delta_shifts=delta_shifts,
            ncc_values=nccs,
            null_params=null_params,
            alpha=stitch_alpha,
            max_shift_pixels=max_shift_pixels,
        )
    )

    zncc_val = None
    zncc_values = None

    # Find all pair-wise overlaps > 0
    fractions_after_stitching, pairs_after_stitching, shifts_after_stitching = (
        find_overlaps(stitched, tile_shape)
    )
    fuse_crop_width = _estimate_crop_width(
        fractions_after_stitching,
        min_overlap_fraction,
        original_tile_shape,
        crop_width,
    )
    logger.info(f"Crop for writing: {fuse_crop_width}.")
    if evaluate_stitching:
        # Global evaluation
        logger.info(
            f"Evaluating stitching quality using {len(pairs_after_stitching):,} overlaps."
        )
        zncc_val, zncc_values = _eval_all_shifts(
            pairs=pairs_after_stitching,
            shifts=shifts_after_stitching,
            read_images=read_images,
        )
        logger.info(f"ZNCC: {zncc_val:.4f}.")

    return dict(
        swap=swap_axes,
        flip_y=flip_y_axis,
        flip_x=flip_x_axis,
        fuse_crop_width=fuse_crop_width,
        max_shift=max_shift,
        align_tile_shape=tile_shape,
        crop_width=crop_width,
        tile_shape=original_tile_shape,
        area_fraction=area_fraction_auto,
        min_overlap_fraction=min_overlap_fraction,
        radial_correction_k=radial_correction_k,
        stitched_positions=stitched,
        pairs=pairs,
        nccs=nccs,
        null_nccs=null_nccs,
        null_params=null_params,
        delta_shifts=delta_shifts,
        zncc_val=zncc_val,
        zncc_values=zncc_values,
        pairs_after_stitching=pairs_after_stitching,
        shifts_after_stitching=shifts_after_stitching,
        fractions_after_stitching=fractions_after_stitching,
        spanning_tree_edges=spanning_tree_edges,
        final_shifts=final_shifts,
        z_threshold=z_threshold,
        valid_edges=valid_edges,
    )
