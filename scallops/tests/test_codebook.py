import numpy as np
import pandas as pd
import pytest
import xarray as xr

from scallops.codebook import (
    _decode_metric,
    _regionprops,
    _regionprops_to_table,
    decode_metric,
    estimate_scale_factors,
    image_to_codes,
)

BASES = ["A", "C", "G", "T"]


@pytest.mark.basecalls
def create_bases_df(reads):
    bases_df = pd.concat(
        [
            _create_read(reads[read_index]).assign(read=read_index + 1)
            for read_index in range(len(reads))
        ]
    )

    bases_df["label"] = 1
    bases_df["tile"] = ""
    bases_df["well"] = ""
    return bases_df


@pytest.mark.basecalls
def _create_read(read):
    data = []
    for cycle in range(len(read)):
        for base in BASES:
            value = 1 if read[cycle] == base else 0
            data.append([base, value, cycle])

    df = pd.DataFrame(data=data, columns=["channel", "intensity", "cycle"])
    return df


@pytest.mark.basecalls
def create_bases_array(reads):
    nreads = len(reads)
    ncycles = len(reads[0])
    data = np.zeros((nreads, ncycles, 4))  # reads, cycles, channels
    for read_index in range(len(reads)):
        read = reads[read_index]
        for cycle in range(len(read)):
            base = read[cycle]
            channel = BASES.index(base)
            data[read_index, cycle, channel] = 1
    return data


@pytest.mark.basecalls
def test_metric_decode():
    """This test exposes 3 test features, each the same normalized trace.

    The first should decode to GENE_A, and pass both the intensity and distance filters The second
    should decode to GENE_B, but fail the intensity filter The third should decode to GENE_B, as it
    is less far from that gene than GENE_A, but should nevertheless fail the distance filter because
    the tiles other than (0, 0) don't match
    """
    values = np.array(
        [
            [[0, 4], [3, 0]],  # this code is decoded "right"
            [[0, 0], [0.4, 0.3]],  # this code should be filtered based on magnitude
            [[30, 0], [0, 40]],  # this code should be filtered based on distance
        ]
    )
    # y+x,t+c
    values = values.reshape(-1, values.shape[1] * values.shape[2])
    codebook = np.array([[[0, 1], [1, 0]], [[0, 0], [1, 1]]])
    codebook = xr.DataArray(codebook, dims=["f", "t", "c"], coords=dict(f=["A", "B"]))
    argmin, distances, values_norms, passes_filter = _decode_metric(
        values, codebook, norm_order=1
    )
    max_distance = 0.5
    min_intensity = 1
    passes_filters = np.logical_and(
        values_norms >= min_intensity, distances <= max_distance, dtype=bool
    )

    features = codebook.f.values[argmin]
    assert np.array_equal(
        features,
        ["A", "B", "B"],
    )
    assert np.array_equal(passes_filters, [True, False, False])


@pytest.mark.basecalls
def test_metric_decode_coords():
    array = xr.DataArray(
        [
            [
                [
                    [10, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 10, 0],
                    [0, 0, 10, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ]
            ],
            [
                [
                    [10, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 10, 0],
                    [0, 0, 10, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ]
            ],
        ],
        dims=["t", "c", "y", "x"],
    )
    codebook = xr.DataArray([[[1], [1]]], dims=["f", "t", "c"])

    codes = image_to_codes(array)

    argmin, distances, trace_norms, passes_filters = _decode_metric(
        array=codes, codebook=codebook, min_intensity=2, norm_order=2
    )
    argmin_, distances_, props = _regionprops(
        argmin, distances, passes_filters, array.sizes
    )
    df = _regionprops_to_table(
        argmin_=argmin_, distances_=distances_, codebook=codebook, props=props
    )
    assert len(df) == 2
    df = df.sort_values("y", ascending=True)
    assert df.iloc[0]["y"] == 0
    assert df.iloc[0]["x"] == 0
    assert df.iloc[0]["area"] == 1

    assert df.iloc[1]["y"] == 2.5
    assert df.iloc[1]["x"] == 2
    assert df.iloc[1]["area"] == 2


@pytest.mark.basecalls
def test_scale_factors():
    values = np.array(
        [
            [[0, 4], [3, 0]],  # this code is decoded "right"
            [[0, 0], [0.4, 0.3]],  # this code should be filtered based on magnitude
            [[30, 0], [0, 40]],  # this code should be filtered based on distance
        ]
    )
    codebook = np.array([[[0, 1], [1, 0]], [[0, 0], [1, 1]]])
    codebook = xr.DataArray(codebook, dims=["f", "t", "c"], coords=dict(f=["A", "B"]))

    scale_factors = estimate_scale_factors(
        xr.DataArray(values.reshape(2, 2, 1, 3), dims=["t", "c", "y", "x"]).expand_dims(
            "z", 2
        ),
        codebook,
        max_distance=0.5,
        min_intensity=1,
        max_iter=1,
        norm_order=1,
        initial_scale_factors=np.ones(4),
    )

    scale_factors_ = np.array([1.0, 1.0, 3 / 3.5, 4 / 3.5])
    assert np.array_equal(scale_factors, scale_factors_)


@pytest.mark.basecalls
def test_scale_factors_multi_iter():
    """Test four simple examples for correct decoding.

    Here the first example should decode to GENE_A, the second to GENE_B. The third is closer to
    GENE_A. The fourth is equidistant to GENE_A and GENE_B, but it picks GENE_A because GENE_A comes
    first in the codebook.
    """
    # (read,t,c)
    values = np.array(
        [
            [[0, 0.5], [0.5, 0]],  # a
            [[0, 0], [0.5, 0.5]],  # b
            [[0.5, 0.5], [0, 0]],  # a
            [[0, 0.5], [0, 0.5]],  # a
        ]
    )

    codebook = np.array([[[0, 1], [1, 0]], [[0, 0], [1, 1]]])
    codebook = xr.DataArray(codebook, dims=["f", "t", "c"], coords=dict(f=["A", "B"]))
    scale_factors = estimate_scale_factors(
        xr.DataArray(values.reshape(2, 2, 2, 2), dims=["t", "c", "y", "x"]).expand_dims(
            "z", 2
        ),
        codebook,
    )
    assert scale_factors[0] == 1
    argmin, distances, values_norms, passes_filter = _decode_metric(
        values.reshape(-1, values.shape[1] * values.shape[2]),
        codebook,
        norm_order=1,
        scale_factors=scale_factors,
    )
    features = codebook.f.values[argmin]
    assert np.array_equal(features[:-1], ["A", "B", "A"])


@pytest.mark.basecalls
def test_simple_intensities_find_correct_nearest_code():
    """Test four simple examples for correct decoding.

    Here the first example should decode to GENE_A, the second to GENE_B. The third is closer to
    GENE_A. The fourth is equidistant to GENE_A and GENE_B, but it picks GENE_A because GENE_A comes
    first in the codebook.
    """
    # (read,t,c)
    values = np.array(
        [
            [[0, 0.5], [0.5, 0]],
            [[0, 0], [0.5, 0.5]],
            [[0.5, 0.5], [0, 0]],
            [[0, 0.5], [0, 0.5]],
        ]
    )
    values = values.reshape(-1, values.shape[1] * values.shape[2])
    codebook = np.array([[[0, 1], [1, 0]], [[0, 0], [1, 1]]])  # A  # B
    codebook = xr.DataArray(codebook, dims=["f", "t", "c"], coords=dict(f=["A", "B"]))
    argmin, distances, values_norms, passes_filter = _decode_metric(
        values, codebook, norm_order=1
    )
    features = codebook.f.values[argmin]
    assert np.array_equal(features, ["A", "B", "A", "A"])


@pytest.mark.basecalls
def test_pixels():
    # t, c, y, x
    values = np.array(
        [
            [[0, 0.5], [0.5, 0]],
            [[0, 0], [0.5, 0.5]],
            [[0.5, 0.5], [0, 0]],
            [[0, 0.5], [0, 0.5]],
        ]
    ).reshape(2, 2, 2, 2)
    codebook = np.array([[[0, 1], [1, 0]], [[0, 0], [1, 1]]])  # A  # B
    codebook = xr.DataArray(codebook, dims=["f", "t", "c"], coords=dict(f=["A", "B"]))
    values = xr.DataArray(values, dims=["t", "c", "y", "x"]).expand_dims("z", 2)
    df = decode_metric(values, codebook, norm_order=1)
    assert len(df) == 2
    assert df.query('feature=="A"').iloc[0].area == 3
    assert df.query('feature=="B"').iloc[0].area == 1
