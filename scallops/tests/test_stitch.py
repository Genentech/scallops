import subprocess

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

from scallops.io import read_image
from scallops.stitch._radial import radial_correct
from scallops.stitch.fuse import _fuse, _fuse_image
from scallops.stitch.utils import tile_overlap_mask, tile_source_labels
from scallops.zarr_io import _write_zarr_image


@pytest.mark.io
def test_radial_correct_multi_channel():
    img = np.arange(100, dtype=np.uint16).reshape((10, 10))
    img2 = radial_correct(img, k=0.001)
    img3 = radial_correct(np.stack((img, img)), k=0.001)
    np.testing.assert_array_equal(img2, img3[0])
    np.testing.assert_array_equal(img2, img3[1])


@pytest.mark.io
def test_tile_overlap_mask():
    rects = [(4, 5), (4, 8), (7, 6), (9, 9)]
    df = pd.DataFrame(data=dict(y=[v[0] for v in rects], x=[v[1] for v in rects]))
    df["distance_to_center"] = np.arange(len(df))
    df["tile"] = ""
    df["source"] = ""
    expected_unfilled = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0],
            [1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        ]
    )
    np.testing.assert_array_equal(
        tile_overlap_mask(df, tile_shape=(4, 4), fill=False),
        expected_unfilled,
        err_msg="Unfilled mask",
    )

    expected_filled = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )
    np.testing.assert_array_equal(
        tile_overlap_mask(df, (4, 4), fill=True), expected_filled, err_msg="Fill mask"
    )


def _write_image_with_position(dest, img, position_y, position_x):
    _write_zarr_image(
        name=None,
        root=dest,
        image=img,
        metadata=dict(
            position_y=position_y,
            position_x=position_x,
            position_x_unit="micrometer",
            position_y_unit="micrometer",
            physical_size_y=1,
            physical_size_x=1,
            physical_size_y_unit="micrometer",
            physical_size_x_unit="micrometer",
        ),
        group=None,
    )


@pytest.mark.io
def test_stitch_preview_cli(tmp_path):
    # ensure it runs
    input_path = tmp_path / "input"

    # top-left, top-right, bottom-left, bottom-right
    coords = [(0, 0), (0.5, 1024 - 50.5), (1000, 0.5), (999, 1024 - 49.5)]
    for i in range(len(coords)):
        c = coords[i]
        img = np.ones((2, 1024, 1024), dtype=np.uint16)
        img[...] = i + 1
        _write_image_with_position(
            input_path / f"test-{i}.zarr",
            xr.DataArray(img, dims=["c", "y", "x"]),
            c[0],
            c[1],
        )

    cmd = [
        "scallops",
        "stitch-preview",
        "--images",
        str(input_path),
        "--image-pattern",
        "{well}-{skip}.zarr",
        "--groupby",
        "well",
        "--output",
        str(tmp_path / "test.tiff"),
    ]

    subprocess.check_call(cmd)


@pytest.mark.io
def test_stitch_cli(tmp_path):
    input_path = tmp_path / "input"

    # top-left, top-right, bottom-left, bottom-right
    coords = [(0, 0), (0.5, 1024 - 50.5), (1000, 0.5), (999, 1024 - 49.5)]
    for i in range(len(coords)):
        c = coords[i]
        img = np.ones((2, 1024, 1024), dtype=np.uint16)
        img[...] = i + 1
        _write_image_with_position(
            input_path / f"test-{i}.zarr",
            xr.DataArray(img, dims=["c", "y", "x"]),
            c[0],
            c[1],
        )

    illum_args = [
        "scallops",
        "illum-corr",
        "agg",
        "--images",
        str(input_path),
        "--image-pattern",
        "{well}-{skip}.zarr",
        "--groupby",
        "well",
        "-o",
        str(tmp_path / "illum"),
    ]
    subprocess.check_call(illum_args)

    test_positions_df = pd.DataFrame(
        data=dict(
            tile=[1, 2, 3, 4],
            value=[1, 2, 3, 4],
            source=[[str(input_path / f"test-{i}.zarr")] for i in range(4)],
            y=[1000, 999.5, 0, 1],
            x=[0.0, 973.5, 0.5, 974.5],
        )
    )

    tile_shape = (1024, 1024)

    coords = test_positions_df[["y", "x"]].values
    center = coords.mean(axis=0)
    test_positions_df["distance_to_center"] = np.sqrt(
        np.sum(coords - center, axis=1) ** 2
    )
    test_positions_df["source_metadata"] = [
        dict(file_metadata=[dict()]) for i in range(len(test_positions_df))
    ]

    test_positions_df = test_positions_df.sort_values(
        by="distance_to_center", ascending=False
    )

    x = test_positions_df["x"].round().values.astype(int)
    y = test_positions_df["y"].round().values.astype(int)
    source = test_positions_df["source"].values
    image_values = test_positions_df["value"].values
    fused_image_shape = (2, 2024, 1998)
    test_fused_sequential = np.zeros(fused_image_shape, dtype=np.uint16)
    expected_value = np.zeros(test_fused_sequential.shape, dtype=np.uint16)

    image_attrs = [dict(file_metadata=[dict()]) for i in range(len(source))]
    for i in range(len(y)):
        expected_value[
            ..., y[i] : y[i] + tile_shape[0], x[i] : x[i] + tile_shape[1]
        ] = image_values[i]
        _fuse_image(
            image_paths=source[i],
            image_attrs=image_attrs[i],
            y=y[i],
            x=x[i],
            blend="none",
            target_shape=test_fused_sequential.shape,
            target_dtype=test_fused_sequential.dtype,
            target=test_fused_sequential,
        )
    np.testing.assert_array_equal(
        test_fused_sequential, expected_value, err_msg="Fused image differs"
    )

    fuse_result = zarr.group()
    _fuse(
        test_positions_df.iloc[[3, 0, 1, 2]],
        blend="none",
        channels_per_batch=1,
        z_index=0,
        group=fuse_result,
    )
    no_blend_array = fuse_result["0"][...].squeeze()
    np.testing.assert_array_equal(
        no_blend_array,
        expected_value,
        err_msg="No blending image differs from expected",
    )
    tiles = tile_source_labels(test_positions_df, tile_shape)
    mask = tiles != 0
    assert (no_blend_array[..., mask] == 0).sum() == 0

    # blending
    stitch_args_blend = [
        "scallops",
        "stitch",
        "--images",
        str(input_path),
        "--image-pattern",
        "{well}-{skip}.zarr",
        "--groupby",
        "well",
        "--image-output",
        str(tmp_path / "stitch-blend.zarr"),
        "--report-output",
        str(tmp_path / "stitch-blend"),
        "--blend",
        "linear",
        "--radial-correction-k",
        "10e-9",
        "--ffp",
        str(tmp_path / "illum" / "test.ome.tiff"),
    ]

    subprocess.check_call(stitch_args_blend)

    blend_fuse_result = zarr.group()
    _fuse(
        pd.read_parquet(tmp_path / "stitch-blend" / "test-positions.parquet"),
        blend="linear",
        channels_per_batch=1,
        radial_correction_k=10e-9,
        crop_width=9,
        z_index=0,
        group=blend_fuse_result,
        ffp=read_image(str(tmp_path / "illum" / "test.ome.tiff")).squeeze(),
    )
    blending_img_cli = read_image(str(tmp_path / "stitch-blend.zarr")).values.squeeze()

    np.testing.assert_array_equal(
        blending_img_cli.squeeze(),
        blend_fuse_result["0"][...].squeeze(),
        err_msg="Blending images differ",
    )


@pytest.mark.io
def test_stitch_crop(tmp_path):
    input_path = tmp_path / "input"

    # top-left, top-right, bottom-left, bottom-right
    coords = [(0, 0), (0.5, 1024 - 50.5), (1000, 0.5), (999, 1024 - 49.5)]
    for i in range(len(coords)):
        c = coords[i]
        img = np.ones((2, 1024, 1024), dtype=np.uint16)
        img[...] = i + 1
        _write_image_with_position(
            input_path / f"test-{i}.zarr",
            xr.DataArray(img, dims=["c", "y", "x"]),
            c[0],
            c[1],
        )

    cmd = [
        "scallops",
        "stitch",
        "--images",
        str(input_path),
        "--image-pattern",
        "{well}-{skip}.zarr",
        "--groupby",
        "well",
        "--image-output",
        str(tmp_path / "stitch.zarr"),
        "--report-output",
        str(tmp_path / "stitch"),
        "--radial-correction-k",
        "none",
        "--crop",
        "50",
    ]

    subprocess.check_call(cmd)

    img = read_image(str(tmp_path / "stitch.zarr")).squeeze().data

    img_shape = img.shape[-2:]
    mask_shape = (
        read_image(str(tmp_path / "stitch.zarr/labels/test-mask")).squeeze().shape
    )
    assert img_shape == mask_shape, f"{img_shape} != {mask_shape}"


@pytest.mark.io
def test_stitch_z_stack(tmp_path):
    input_path = tmp_path / "input"

    # top-left, top-right, bottom-left, bottom-right
    coords = [(0, 0), (0, 100), (100, 0), (100, 100)]
    constant_img_val = 10000
    for i in range(len(coords)):
        c = coords[i]

        for z_index in range(2):
            if z_index == 0:  # constant large value will be selected for max projection
                img = np.zeros((1, 100, 100), dtype=np.uint16)
                img[...] = constant_img_val
            else:  # will be selected for best focus
                img = np.arange(100 * 100, dtype=np.uint16).reshape(1, 100, 100)
            _write_image_with_position(
                input_path / f"test-tile{i}-z{z_index}.zarr",
                xr.DataArray(img, dims=["c", "y", "x"]),
                c[0],
                c[1],
            )
    cmd = [
        "scallops",
        "stitch",
        "--images",
        str(input_path),
        "--image-pattern",
        "{well}-tile{tile}-z{z}.zarr",
        "--groupby",
        "well",
        "--image-output",
        str(tmp_path / "max.zarr"),
        "--report-output",
        str(tmp_path / "max"),
        "--radial-correction-k",
        "none",
        "--z-index",
        "max",
        "--no-save-labels",
    ]

    subprocess.check_call(cmd)
    cmd = [
        "scallops",
        "stitch",
        "--images",
        str(input_path),
        "--image-pattern",
        "{well}-tile{tile}-z{z}.zarr",
        "--groupby",
        "well",
        "--image-output",
        str(tmp_path / "focus.zarr"),
        "--report-output",
        str(tmp_path / "focus"),
        "--radial-correction-k",
        "none",
        "--z-index",
        "focus",
        "--no-save-labels",
    ]

    subprocess.check_call(cmd)

    df_positions_z = pd.read_parquet(str(tmp_path / "max" / "test-positions.parquet"))
    assert "z_index" not in df_positions_z.columns
    for s in df_positions_z["source"].values:  # should have all z to take max
        assert len(s) == 2, s

    df_positions_focus = pd.read_parquet(
        str(tmp_path / "focus" / "test-positions.parquet")
    )
    for s in df_positions_focus["source"].values:  # should only have selected z
        assert s[0].endswith("-z1.zarr")
        assert len(s) == 1, s
    assert all(df_positions_focus["z_index"] == 1)
    max_img = read_image(str(tmp_path / "max.zarr")).squeeze().data
    assert np.all(max_img == constant_img_val)
    focus_img = read_image(str(tmp_path / "focus.zarr")).squeeze().data
    assert np.all(focus_img < constant_img_val)
