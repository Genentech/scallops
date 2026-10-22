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
    check_channel_order,
    check_stage_positions_only,
    edge_band_width,
    fusion_crop,
    load_params,
    metadata_channel_names,
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
    order = ["CY7", " tl", "dapi", "FITC"]  # case and surrounding spaces don't matter
    r, est = resolve_channel_alignment("all", channel_order=order, **kw)
    assert sorted(r) == [0, 1, 3] and est == []
    assert r[1]["m"] == 0.00075 and r[1]["same_as"] == "FITC"
    assert r[0]["ty"] == 0.95
    # the reference is skipped, channels outside the output are skipped
    r, est = resolve_channel_alignment(
        [0, 2], channel_order=order, **dict(kw, output_channels=[1, 2, 3])
    )
    assert r == {} and est == []
    # no entry ('-' or not in the file): estimated; no parameter file: every requested channel is estimated
    r, est = resolve_channel_alignment(
        [0, 1, 3], channel_order=["CY7", "-", "DAPI", "CY5"], **kw
    )
    assert sorted(r) == [0] and est == [1, 3]
    r, est = resolve_channel_alignment(
        "all", channel_order=order, **dict(kw, params=None)
    )
    assert r == {} and est == [0, 1, 3]
    # wrong length, wrong reference, wrong tile shape, out of range
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


def test_channel_names_are_text_and_case_insensitive(tmp_path):
    doc = _params_doc(
        **{"405": {"m": 0.0, "ty": 0.0, "tx": 0.0}, "Cy7": {"same_as": 405}}
    )
    doc["reference_channel"] = 405
    params = load_params(_write_params(tmp_path / "p.json", doc))
    assert (
        params["reference_channel"] == "405"
        and params["channels"]["cy7"]["same_as"] == "405"
    )
    r, est = resolve_channel_alignment(
        "all",
        params=params,
        channel_order=[405, "CY7"],
        n_channels=2,
        reference_channel=0,
        output_channels=None,
        tile_shape=(TILE, TILE),
    )
    assert sorted(r) == [1] and est == [] and r[1]["name"] == "CY7"


def test_check_channel_order_warns_on_mismatch(caplog):
    check_channel_order(["DAPI", "cy7", "-", "FITC"], ["dapi", "CY5", "TRITC", None])
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1 and "Channel 1" in warnings[0] and "CY5" in warnings[0]


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


def _write_tiles(path, rng, displaced, m=M, ty=TY, tx=TX, positions=POSITIONS):
    """Channel 0: blobs on a global grid; channel 1: the same blobs, displaced in each tile by the inverse of
    s(p) = c + (1 + m)(p - c) + (ty, tx), so that the alignment correction maps channel 1 back onto channel 0."""
    centres = rng.uniform(0, 2 * TILE - 64, (1200, 2))
    c = TILE / 2
    yy, xx = np.mgrid[0:TILE, 0:TILE].astype(float)
    for i, (py, px) in enumerate(positions):
        ch0 = _blobs(yy + py, xx + px, centres)
        if displaced:
            qy, qx = c + (yy - ty - c) / (1 + m), c + (xx - tx - c) / (1 + m)
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
            metadata_name=None,
            m=M,
            ty=TY,
            tx=TX,
            same_as=None,
            source="params_file",
            estimate=None,
        )
    ]
    assert [c["name"] for c in record["channel_identification"]] == ["DAPI", "CY7"]
    assert record["estimation"] is None
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


def test_bare_correction_channels_flag_is_an_error(tmp_path):
    r = subprocess.run(
        [
            "scallops",
            "stitch",
            "--images",
            str(tmp_path),
            "--report-output",
            str(tmp_path / "stitch"),
            "--alignment-correction-channels",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 2 and "expected at least one argument" in r.stderr


def test_metadata_channel_names_open_a_multichannel_file_once(tmp_path, monkeypatch):
    import tifffile

    import scallops.io

    names = ["DAPI", "FITC", "TRITC", "CY5", "CY7"]
    path = tmp_path / "five.ome.tif"
    tifffile.imwrite(
        path,
        np.zeros((5, 16, 16), np.uint16),
        ome=True,
        metadata={"axes": "CYX", "Channel": {"Name": names}},
    )
    opens = []
    create_image = scallops.io._create_image
    monkeypatch.setattr(
        scallops.io,
        "_create_image",
        lambda p, **kw: opens.append(p) or create_image(p, **kw),
    )
    assert metadata_channel_names([str(path)], None, 5) == names
    assert len(opens) == 1


def test_estimation_reads_the_z_plane_fusion_uses(tmp_path, monkeypatch):
    import tifffile

    import scallops.stitch._stitch as S
    from scallops.stitch.fuse import _fuse_image

    # 3 z planes in one file, each plane a constant: the value read identifies the plane
    data = np.zeros((3, 2, 32, 32), np.uint16)
    for z in range(3):
        data[z] = 1000 * (z + 1)
    path = tmp_path / "zstack.ome.tif"
    tifffile.imwrite(path, data, ome=True, metadata={"axes": "ZCYX"})

    class Stop(Exception):
        pass

    captured = {}

    def capture_reference(read_tile, *args, **kwargs):
        captured["ref"] = read_tile(0, [])[0]
        raise Stop

    monkeypatch.setattr(S, "estimate_well", capture_reference)
    # one z for all tiles, and per-tile z values as --z-index focus gives them (Python ints)
    for z_index, fusion_z in ((1, 1), ((1,), np.asarray((1,))[0])):
        with pytest.raises(Stop):
            S._estimate_channel_alignment(
                to_estimate=[1],
                channel_order=["DAPI", "CY7"],
                reference_channel=0,
                filepaths=[[str(path)]],
                fileattrs=None,
                n_scenes=None,
                stage_positions=np.zeros((1, 2)),
                image_spacing=None,
                z_index=z_index,
                t_index=None,
                random_seed=0,
                image_key="A1",
            )
        # fusion gets per-tile z values from the positions table (numpy integers)
        target = np.zeros((1, 32, 32), np.uint16)
        _fuse_image(
            [str(path)],
            None,
            0,
            0,
            (32, 32),
            target,
            np.uint16,
            output_channels=[0],
            z_index=fusion_z,
        )
        np.testing.assert_array_equal(captured["ref"], target[0])


def test_edge_band_width():
    assert edge_band_width((512, 512), None) == (0, 0)
    assert edge_band_width((512, 512), None, 0.0, 30.0, 0.0) == (30, 0)
    assert edge_band_width((512, 512), None, 0.0, -12.5, 7.2) == (13, 8)


def test_fusion_crop_covers_zero_bands_or_stops():
    def shifted(ty):
        return {1: dict(name="CY7", m=0.0, ty=ty, tx=0.0)}

    used, rec = fusion_crop(shifted(10.0), (512, 512), None, (24, 24), (32, 32), "A1")
    assert used == (24, 24)
    used, rec = fusion_crop(shifted(30.0), (512, 512), None, (24, 24), (32, 32), "A1")
    assert used == (30, 24) and rec == dict(
        scallops=[24, 24], edge_band=[30, 0], used=[30, 24], max=[32, 32]
    )
    with pytest.raises(
        ValueError,
        match=r"Tile overlap is too small to correct channel 1 \(CY7\) of A1: .* 40 px band .* only 32 px",
    ):
        fusion_crop(shifted(40.0), (512, 512), None, (24, 24), (32, 32), "A1")
    # a band inside SCALLOPS's own crop is accepted even where the overlap limit is smaller
    assert fusion_crop(shifted(30.0), (512, 512), None, (32, 24), (20, 20), "A1")[
        0
    ] == (32, 24)
    # no neighbours along an axis (e.g. a single tile): any crop opens no gaps
    assert fusion_crop(shifted(40.0), (512, 512), None, (0, 0), (None, None), "A1")[
        0
    ] == (40, 0)


def _shifted_stitch(tmp_path, ty, positions=POSITIONS):
    _write_tiles(
        tmp_path / "input",
        np.random.default_rng(1),
        True,
        m=0.0,
        ty=ty,
        tx=0.0,
        positions=positions,
    )
    params = _write_params(
        tmp_path / "params.json", _params_doc(CY7={"m": 0.0, "ty": ty, "tx": 0.0})
    )
    return _stitch_cmd(
        tmp_path,
        "--alignment-correction-channels",
        "all",
        "--alignment-correction-params",
        params,
        "--alignment-correction-channel-order",
        "DAPI",
        "CY7",
    )


@pytest.mark.io
def test_stitch_crop_removes_zero_bands(tmp_path):
    # 64 px overlap: SCALLOPS's crop is 24 px, half the overlap 32 px; a 30 px shift leaves a 30 px band
    subprocess.check_call(_shifted_stitch(tmp_path, 30.0))
    fused = read_image(str(tmp_path / "stitch.zarr")).values
    assert (fused[1] == 0).sum() == 0
    record = json.loads((tmp_path / "stitch" / "A1-channel-alignment.json").read_text())
    crop = record["fusion_crop"]
    assert crop["scallops"] == [24, 24] and crop["edge_band"] == [30, 0]
    assert crop["used"] == [30, 24] and min(crop["max"]) >= 30


@pytest.mark.io
def test_stitch_stops_when_overlap_is_too_small(tmp_path):
    r = subprocess.run(_shifted_stitch(tmp_path, 40.0), capture_output=True, text=True)
    assert r.returncode != 0
    assert (
        "Tile overlap is too small to correct channel 1 (CY7) of A1"
        in r.stdout + r.stderr
    )


def test_max_fusion_crop_counts_side_by_side_neighbours():
    from scallops.stitch._stitch import _max_fusion_crop

    # 2 x 2 grid, 64 px overlaps; one pair 12 px further apart in x; the diagonal pairs don't count
    positions = [(0, 0), (0, 460), (448, 0), (448, 448)]
    pairs = [(0, 1), (0, 2), (1, 3), (2, 3), (0, 3), (1, 2)]
    result = dict(
        stitched_positions=positions, pairs_after_stitching=pairs, tile_shape=(512, 512)
    )
    # typical overlap per axis: y pairs overlap 64 and 64 px, x pairs 52 and 64 px (median 58)
    assert _max_fusion_crop(result) == (32, 29)
    # a single tile: no neighbours, any crop opens no gaps
    assert _max_fusion_crop(
        dict(result, stitched_positions=[(0, 0)], pairs_after_stitching=None)
    ) == (None, None)


def test_estimators_survive_patches_leaving_the_tile(monkeypatch):
    import scallops.stitch._channel_align_estimate as E

    rng = np.random.default_rng(2)
    shape = (256, 256)
    centres = rng.uniform(80, 176, (40, 2))
    settings = E.scale_settings(9.0)
    cp = E._prep(rng.normal(300, 30, shape))
    # a translation that moves every patch out of the tile: not converged, rather than an error
    *_, step = E.method_et(
        centres, cp, shape, settings, dict(m=0.0, ty=500.0, tx=0.0), 9.0
    )
    assert step == float("inf")
    # S's own estimate moving every patch out of the tile: fails the rms rule, rather than an error
    monkeypatch.setattr(E, "_sym_shift", lambda P: np.array([500.0, 0.0]))
    _, rms, _ = E.method_s(centres, cp, shape, settings)
    assert rms == float("inf")


def test_stage_positions_only_stops_when_bands_would_remain():
    alignment = {
        1: dict(name="FITC", m=0.0, ty=1.5, tx=0.0),
        2: dict(name="CY7", m=0.0, ty=0.0, tx=0.0),
        3: dict(name="CY5", m=0.0, ty=0.0, tx=-2.0),
    }
    check_stage_positions_only(
        {2: alignment[2]}, (512, 512), None, "A1"
    )  # no band: corrected
    with pytest.raises(ValueError) as err:
        check_stage_positions_only(alignment, (512, 512), None, "A1")
    message = str(err.value)
    assert message.startswith("Tiles of A1 could not be aligned")
    assert (
        "channel(s) 1 (FITC), 3 (CY5) would leave bands" in message
        and "CY7" not in message
    )
    assert (
        "Remove only these channel(s) from --alignment-correction-channels" in message
    )


@pytest.mark.io
def test_single_tile_crop_removes_zero_band(tmp_path):
    subprocess.check_call(_shifted_stitch(tmp_path, 30.0, positions=[(0, 0)]))
    fused = read_image(str(tmp_path / "stitch.zarr")).values
    assert (fused[1] == 0).sum() == 0
    record = json.loads((tmp_path / "stitch" / "A1-channel-alignment.json").read_text())
    assert record["fusion_crop"]["used"] == [30, 0]


@pytest.mark.io
def test_stitch_stops_when_tiles_are_placed_at_stage_positions(tmp_path):
    # tiles 88 px apart: no overlaps, so SCALLOPS places them at their stage positions without cropping
    spaced = [(0, 0), (0, TILE + 88), (TILE + 88, 0), (TILE + 88, TILE + 88)]
    r = subprocess.run(
        _shifted_stitch(tmp_path, 3.0, positions=spaced), capture_output=True, text=True
    )
    assert r.returncode != 0
    assert "were placed at their stage positions" in r.stdout + r.stderr
