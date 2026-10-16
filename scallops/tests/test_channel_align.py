import json
import subprocess

import numpy as np
import pyarrow.parquet as pq
import pytest
import xarray as xr
from skimage.registration import phase_cross_correlation

from scallops.io import read_image
from scallops.stitch._channel_align import (
    alignment_coords,
    load_params,
    parse_correction_channels,
    resolve_channel_alignment,
    warp_image,
)
from scallops.stitch._radial import _radial_mapping
from scallops.zarr_io import _write_zarr_image

TILE = 512
# (y, x) stage positions in µm (pixel size 1 µm): a 2 x 2 grid with 64 px overlaps
POSITIONS = [(0, 0), (0, TILE - 64), (TILE - 64, 0), (TILE - 64, TILE - 64)]
M, TY, TX = 0.004, 2.3, -1.7


def _params_doc(**channels):
    return {
        "format": "scallops-channel-alignment",
        "version": 1,
        "tile_shape": [TILE, TILE],
        "reference_channel": "DAPI",
        "channels": channels,
    }


def _write_params(path, doc):
    path.write_text(json.dumps(doc))
    return str(path)


def test_parse_correction_channels():
    assert parse_correction_channels(None) is None
    assert parse_correction_channels([]) is None
    assert parse_correction_channels(["3", "1", "3"]) == [1, 3]
    for token in ("all", "ALL", "*", "true", "True"):
        assert parse_correction_channels([token]) == "all"
    assert parse_correction_channels(True) == "all"
    with pytest.raises(ValueError):
        parse_correction_channels(["all", "1"])
    with pytest.raises(ValueError):
        parse_correction_channels(["FITC"])
    with pytest.raises(ValueError):
        parse_correction_channels(["-1"])


def test_load_params(tmp_path):
    doc = _params_doc(
        FITC={"m": 0.00075, "ty": 0.34, "tx": -0.16}, TL={"same_as": "FITC"}
    )
    p = load_params(_write_params(tmp_path / "p.json", doc))
    assert p["tile_shape"] == (TILE, TILE)
    assert p["channels"]["tl"]["m"] == 0.00075
    assert p["channels"]["tl"]["same_as"] == "FITC"
    assert len(p["sha256"]) == 64

    bad = [
        dict(doc, format="other"),
        dict(doc, tile_shape=[TILE]),
        _params_doc(FITC={"m": 0.1, "ty": 0.0}),
        _params_doc(FITC={"m": float("nan"), "ty": 0.0, "tx": 0.0}),
        _params_doc(TL={"same_as": "FITC"}),
        _params_doc(
            FITC={"m": 0, "ty": 0, "tx": 0}, A={"same_as": "FITC"}, B={"same_as": "A"}
        ),
        _params_doc(FITC={"m": 0, "ty": 0, "tx": 0}, fitc={"m": 0, "ty": 0, "tx": 0}),
    ]
    for i, d in enumerate(bad):
        with pytest.raises(ValueError):
            load_params(_write_params(tmp_path / f"bad{i}.json", d))


def test_resolve_channel_alignment(tmp_path):
    doc = _params_doc(
        FITC={"m": 0.00075, "ty": 0.34, "tx": -0.16},
        CY7={"m": -0.0015, "ty": 0.95, "tx": 0.3},
        TL={"same_as": "FITC"},
    )
    params = load_params(_write_params(tmp_path / "p.json", doc))
    kw = dict(
        params=params,
        n_channels=4,
        reference_channel=2,
        output_channels=None,
        tile_shape=(TILE, TILE),
    )
    order = ["CY7", "tl", "dapi", "FITC"]
    r = resolve_channel_alignment("all", channel_order=order, **kw)
    assert sorted(r) == [0, 1, 3]
    assert r[1]["m"] == 0.00075 and r[1]["same_as"] == "FITC"
    assert r[0]["ty"] == 0.95
    # the reference is skipped, channels outside the output are skipped
    r = resolve_channel_alignment(
        [0, 2], channel_order=order, **dict(kw, output_channels=[1, 2, 3])
    )
    assert r == {}
    # missing entry, wrong length, wrong reference, wrong tile shape, out of range
    with pytest.raises(ValueError, match="No entry"):
        resolve_channel_alignment([1], channel_order=["CY7", "-", "DAPI", "FITC"], **kw)
    with pytest.raises(ValueError, match="has 3 names"):
        resolve_channel_alignment("all", channel_order=order[:3], **kw)
    with pytest.raises(ValueError, match="reference channel"):
        resolve_channel_alignment(
            "all", channel_order=order, **dict(kw, reference_channel=0)
        )
    with pytest.raises(ValueError, match="tiles of"):
        resolve_channel_alignment(
            "all", channel_order=order, **dict(kw, tile_shape=(10, 10))
        )
    with pytest.raises(ValueError, match="out of range"):
        resolve_channel_alignment([5], channel_order=order, **kw)
    with pytest.raises(ValueError, match="required"):
        resolve_channel_alignment("all", channel_order=None, **kw)


def test_alignment_coords_match_radial_mapping():
    shape, k = (64, 80), 1e-5
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    xy = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    expected = _radial_mapping(xy, np.array(shape) / 2, k)
    coords = alignment_coords(shape, k)
    np.testing.assert_allclose(coords[1].ravel(), expected[:, 0], atol=1e-4)
    np.testing.assert_allclose(coords[0].ravel(), expected[:, 1], atol=1e-4)
    assert alignment_coords(shape, None) is None
    # alignment only: s(p) = c + (1 + m)(p - c) + t
    c = np.array(shape) / 2
    a = alignment_coords(shape, None, M, TY, TX)
    np.testing.assert_allclose(a[0], c[0] + (1 + M) * (yy - c[0]) + TY, atol=1e-4)
    np.testing.assert_allclose(a[1], c[1] + (1 + M) * (xx - c[1]) + TX, atol=1e-4)


def _blobs(y, x, centres, sigma=2.5):
    """Analytic image: sum of Gaussian blobs, evaluated at arbitrary coordinates."""
    out = np.zeros(np.broadcast(y, x).shape)
    for cy, cx in centres:
        out += np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma**2))
    return out


def test_warp_image_accuracy():
    rng = np.random.default_rng(0)
    shape = (128, 128)
    centres = rng.uniform(10, 118, (40, 2))
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(float)
    raw = _blobs(yy, xx, centres) / 4
    coords = alignment_coords(shape, 2e-6, M, TY, TX)
    out = warp_image(raw, coords)
    expected = _blobs(coords[0], coords[1], centres) / 4
    inner = (slice(12, -12), slice(12, -12))
    assert np.abs(out - expected)[inner].max() < 0.01 * expected.max()
    # integer input is scaled like skimage's img_as_float
    out16 = warp_image((raw * 60000).astype(np.uint16), coords)
    assert out16.max() <= 1.0


def _write_tiles(path, rng, displaced):
    """Channel 0: blobs on a global grid; channel 1: the same blobs, displaced in each tile by the inverse of
    s(p) = c + (1 + M)(p - c) + (TY, TX), so that the alignment correction maps channel 1 back onto channel 0."""
    centres = rng.uniform(0, 2 * TILE - 64, (1200, 2))
    c = TILE / 2
    yy, xx = np.mgrid[0:TILE, 0:TILE].astype(float)
    for i, (py, px) in enumerate(POSITIONS):
        ch0 = _blobs(yy + py, xx + px, centres)
        if displaced:
            qy, qx = c + (yy - TY - c) / (1 + M), c + (xx - TX - c) / (1 + M)
        else:
            qy, qx = yy, xx
        ch1 = _blobs(qy + py, qx + px, centres)
        img = (np.stack([ch0, ch1]) * 20000 + 500).clip(0, 65535).astype(np.uint16)
        _write_zarr_image(
            name=None,
            root=path / f"A1-{i}.zarr",
            image=xr.DataArray(img, dims=["c", "y", "x"]),
            metadata=dict(
                position_y=py,
                position_x=px,
                position_x_unit="micrometer",
                position_y_unit="micrometer",
                physical_size_y=1,
                physical_size_x=1,
                physical_size_y_unit="micrometer",
                physical_size_x_unit="micrometer",
            ),
            group=None,
        )


def _stitch_cmd(tmp_path, *extra):
    return [
        "scallops",
        "stitch",
        "--images",
        str(tmp_path / "input"),
        "--image-pattern",
        "{well}-{skip}.zarr",
        "--groupby",
        "well",
        "--image-output",
        str(tmp_path / "stitch.zarr"),
        "--report-output",
        str(tmp_path / "stitch"),
        "--no-evaluate",
        "--radial-correction-k",
        "none",
        *extra,
    ]


@pytest.mark.io
def test_stitch_alignment_correction(tmp_path):
    _write_tiles(tmp_path / "input", np.random.default_rng(1), displaced=True)
    params = _write_params(
        tmp_path / "params.json", _params_doc(CY7={"m": M, "ty": TY, "tx": TX})
    )
    subprocess.check_call(
        _stitch_cmd(
            tmp_path,
            "--alignment-correction-channels",
            "all",
            "--alignment-correction-params",
            params,
            "--alignment-correction-channel-order",
            "DAPI",
            "CY7",
        )
    )
    fused = read_image(str(tmp_path / "stitch.zarr")).values.astype(float)
    inner = (slice(40, -40), slice(40, -40))
    shift, _, _ = phase_cross_correlation(
        fused[0][inner], fused[1][inner], upsample_factor=20
    )
    assert np.abs(shift).max() < 0.1, shift
    r = np.corrcoef(fused[0][inner].ravel(), fused[1][inner].ravel())[0, 1]
    assert r > 0.99, r

    record = json.loads((tmp_path / "stitch" / "A1-channel-alignment.json").read_text())
    assert record["image"] == "A1"
    assert record["reference_channel"]["index"] == 0
    assert record["channels"] == [
        dict(
            index=1,
            name="CY7",
            channel_name=None,
            m=M,
            ty=TY,
            tx=TX,
            same_as=None,
            source="params_file",
        )
    ]
    meta = json.loads(
        pq.read_schema(tmp_path / "stitch" / "A1-positions.parquet").metadata[
            b"scallops"
        ]
    )
    assert meta["channel_alignment"]["channels"][0]["ty"] == TY


@pytest.mark.io
def test_stitch_without_alignment_correction_is_unchanged(tmp_path):
    _write_tiles(tmp_path / "input", np.random.default_rng(1), displaced=True)
    subprocess.check_call(_stitch_cmd(tmp_path))
    assert not (tmp_path / "stitch" / "A1-channel-alignment.json").exists()
    meta = json.loads(
        pq.read_schema(tmp_path / "stitch" / "A1-positions.parquet").metadata[
            b"scallops"
        ]
    )
    assert "channel_alignment" not in meta
    fused = read_image(str(tmp_path / "stitch.zarr")).values.astype(float)
    inner = (slice(40, -40), slice(40, -40))
    shift, _, _ = phase_cross_correlation(
        fused[0][inner], fused[1][inner], upsample_factor=20
    )
    assert np.abs(shift).max() > 1  # channel 1 is still displaced


@pytest.mark.io
def test_stitch_alignment_correction_refuses_channel_reference(tmp_path):
    _write_tiles(tmp_path / "input", np.random.default_rng(1), displaced=False)
    params = _write_params(
        tmp_path / "params.json", _params_doc(CY7={"m": M, "ty": TY, "tx": TX})
    )
    result = subprocess.run(
        _stitch_cmd(
            tmp_path,
            "--alignment-correction-channels",
            "1",
            "--alignment-correction-params",
            params,
            "--alignment-correction-channel-order",
            "DAPI",
            "CY7",
            "--channel-reference",
            "0",
        ),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--channel-reference" in result.stderr
