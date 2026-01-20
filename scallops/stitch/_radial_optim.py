"""Find optimal K for radial correction"""

import dask
import numpy as np
from dask import delayed

from scallops.stitch._radial import radial_correct
from scallops.stitch.shift_utils import calc_best_shift


def _sample_random_pairs(pairs, size=21, seed=239753):
    # Make sure size is odd so that median is one image (not the average of two)
    if size % 2 == 0:
        size -= 1
    rng = np.random.default_rng(seed=seed)
    pair_vec = rng.choice(pairs.shape[0], size=size, replace=False)
    return pair_vec


def parallel_find_radial_K(pair_vec, read_images, pairs, shifts, upsample_factor=1.0):
    results = []
    _find_radial_K_delayed = delayed(radial_best_K, nout=3)
    for pos in pair_vec:
        ref = read_images[pairs[pos][0]]
        mov = read_images[pairs[pos][1]]
        results.append(
            _find_radial_K_delayed(
                ref,
                mov,
                shifts[pos],
                upsample_factor=upsample_factor,
            )
        )
    results = np.array(
        list(dask.compute(*results)),
        dtype=[("K", float), ("nccv", float), ("crop_width", int)],
    )
    assert pair_vec.size % 2 == 1  # Make sure the size is odd
    id_median = np.argsort(results["K"])[pair_vec.size // 2]
    return results[id_median]["K"], results[id_median]["crop_width"]


grids = [-1e-7, -1e-8, -1e-9, 0.0, 1e-9, 1e-8, 1e-7]
tildes = [1e-8, 1e-9, 1e-10, 1e-10, 1e-10, 1e-9, 1e-8]


def _radial_crop_width(img):
    tmp_ = img > 0
    width = int(
        ((tmp_.sum(axis=0) == 0).sum() + (tmp_.sum(axis=1) == 0).sum()) / 4.0 + 0.5
    )  # round to nearest integer
    return width


def _radial_get_ncc(ref, mov, K, orig_shift, overlap_min, upsample_factor):
    refc = radial_correct(ref, K)
    movc = radial_correct(mov, K)
    cropw = max(_radial_crop_width(refc), _radial_crop_width(movc))

    if (
        np.abs(orig_shift).max() + 10 > ref.shape[0] - cropw * 2
    ):  # at least 10 pixel overlap
        return 0.0, cropw

    if cropw > 0:
        refc = refc[cropw:-cropw, cropw:-cropw]
        movc = movc[cropw:-cropw, cropw:-cropw]

    _, ncc_value = calc_best_shift(
        refc, movc, orig_shift, upsample_factor=upsample_factor, overlap_min=overlap_min
    )
    return ncc_value, cropw


def _radial_neighbor_search(
    ref, mov, K, tilde, orig_shift, overlap_min, upsample_factor
):
    triplet = np.zeros(3, dtype=[("ncc", float), ("cropw", int)])
    triplet[0] = _radial_get_ncc(
        ref, mov, K, orig_shift, overlap_min, upsample_factor
    )  # Center
    if tilde > 0.0:
        triplet[1] = _radial_get_ncc(
            ref, mov, K - tilde, orig_shift, overlap_min, upsample_factor
        )  # Left
        triplet[2] = _radial_get_ncc(
            ref, mov, K + tilde, orig_shift, overlap_min, upsample_factor
        )  # Right
    return triplet["ncc"].max(), triplet


def _radial_coarse_scan(ref, mov, orig_shift, overlap_min, upsample_factor):
    max_ncc = -np.inf
    best_pos = -1
    best_triplet = None
    for pos, (K, tilde) in enumerate(zip(grids, tildes)):
        ncc_value, triplet = _radial_neighbor_search(
            ref, mov, K, tilde, orig_shift, overlap_min, upsample_factor
        )
        if max_ncc < ncc_value:
            max_ncc = ncc_value
            best_pos = pos
            best_triplet = triplet

    return max_ncc, best_pos, best_triplet


def _radial_detail_scan(
    ref, mov, orig_shift, nccv, posv, orig_triplet, overlap_min, upsample_factor
):
    if posv == 3:  # K = 0.0
        detail_Ks = [-5e-10, -4e-10, -3e-10, -2e-10, 2e-10, 3e-10, 4e-10, 5e-10]
        detail_tildes = [0.0] * 8
    else:
        detail_Ks = [
            grids[posv] - np.sign(grids[posv]) * i * tildes[posv] for i in range(1, 5)
        ]
        detail_tildes = [tildes[posv] / 10.0 if tildes[posv] > 1e-10 else 0.0] * 4
        detail_Ks += [grids[posv] * i for i in range(2, 6)]
        detail_tildes += [tildes[posv]] * 4

    max_ncc = nccv
    best_K = grids[posv]
    best_tilde = tildes[posv]
    best_triplet = orig_triplet
    for K, tilde in zip(detail_Ks, detail_tildes):
        ncc_value, triplet = _radial_neighbor_search(
            ref, mov, K, tilde, orig_shift, overlap_min, upsample_factor
        )
        if max_ncc < ncc_value:
            max_ncc = ncc_value
            best_K = K
            best_tilde = tilde
            best_triplet = triplet

    return max_ncc, best_K, best_tilde, best_triplet


def _radial_final_scan(
    ref, mov, orig_shift, K, tilde, orig_triplet, overlap_min, upsample_factor
):
    res_vec = np.zeros(9, dtype=[("ncc", float), ("cropw", int)])
    Ks = np.zeros(9)
    res_vec[0:3] = orig_triplet
    pos = 0
    Ks[0] = K
    for i in range(1, 5):
        pos += 1
        Ks[pos] = K - i * tilde
        if i > 1:
            res_vec[pos] = _radial_get_ncc(
                ref, mov, Ks[pos], orig_shift, overlap_min, upsample_factor
            )
        pos += 1
        Ks[pos] = K + i * tilde
        if i > 1:
            res_vec[pos] = _radial_get_ncc(
                ref, mov, Ks[pos], orig_shift, overlap_min, upsample_factor
            )
    best = np.argmax(res_vec["ncc"])

    return Ks[best], res_vec[best][0], res_vec[best][1]


def radial_best_K(ref, mov, orig_shift, overlap_min=0.01, upsample_factor=1):
    max_ncc, best_pos, best_triplet = _radial_coarse_scan(
        ref, mov, orig_shift, overlap_min, upsample_factor
    )
    max_ncc, best_K, best_tilde, best_triplet = _radial_detail_scan(
        ref,
        mov,
        orig_shift,
        max_ncc,
        best_pos,
        best_triplet,
        overlap_min,
        upsample_factor,
    )
    if best_tilde >= 1e-10:
        best_K, max_ncc, crop_width = _radial_final_scan(
            ref,
            mov,
            orig_shift,
            best_K,
            best_tilde,
            best_triplet,
            overlap_min,
            upsample_factor,
        )
    else:
        crop_width = best_triplet["cropw"][0]

    return best_K, max_ncc, crop_width
