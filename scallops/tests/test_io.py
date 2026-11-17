import argparse
import glob
import os
import shutil

import anndata
import dask.array as da
import dask.dataframe as dd
import numpy as np
import ome_types
import pandas as pd
import pytest
import skimage
import xarray as xr
import zarr
from dask import delayed
from ome_zarr.format import FormatV04, FormatV05
from zarr.core.group import AsyncGroup
from zarr.errors import ContainsGroupError

from scallops import zarr_io
from scallops.cli.args import _label_args_type
from scallops.cli.util import (
    _group_src_attrs,
    find_label_group,
    last_label_match,
)
from scallops.datasets._registry import cache_dir
from scallops.experiment.elements import Experiment
from scallops.features.util import _slice_anndata
from scallops.io import (
    _dd_to_parquet,
    _images2fov,
    _match_size,
    _set_up_experiment,
    get_image_spacing,
    is_anndata,
    is_parquet_file,
    is_scallops_zarr,
    read_anndata,
    read_experiment,
    read_image,
    save_ome_tiff,
    to_image_montage,
    write_anndata_zarr,
)
from scallops.zarr_io import (
    _add_metadata,
    _current_format,
    _ome_attrs,
    _physical_pixel_size,
    _require_group,
    _write_zarr_image,
    _write_zarr_labels,
    open_ome_zarr,
    read_ome_zarr_array,
)

DATA_DIR = cache_dir()


@pytest.mark.io
def test_is_scallops_zarr(tmp_path):
    data = anndata.AnnData(
        X=np.ones((2, 2)),
        obs=pd.DataFrame(index=["1", "2"]),
        var=pd.DataFrame(index=["1", "2"]),
    )
    path = os.path.join(tmp_path, "test.zarr")
    write_anndata_zarr(data, path)

    assert not is_scallops_zarr(path)
    store = zarr.open(path, mode="r+")
    store.attrs["scallops"] = "test"
    assert is_scallops_zarr(path)
    assert anndata.read_zarr(path).shape == (2, 2)


@pytest.mark.io
def test_is_anndata_zarr(tmp_path):
    d = anndata.AnnData(
        X=np.ones((2, 2)),
    )
    path1 = tmp_path / "test1.zarr"
    write_anndata_zarr(d, path1, convert_strings_to_categoricals=False)
    assert is_anndata(path1)

    @delayed
    def create_array(fail):
        if fail:
            raise ValueError("fail")
        return np.ones((3, 3), dtype=int)

    path2 = tmp_path / "test2.zarr"
    X = da.concatenate(
        [
            da.from_delayed(create_array(False), shape=(3, 3), dtype=int),
            da.from_delayed(create_array(True), shape=(3, 3), dtype=int),
        ]
    )
    d = anndata.AnnData(X=X)
    try:
        write_anndata_zarr(d, path2)
    except ValueError:
        pass
    assert not is_anndata(path2)


@pytest.mark.io
def test_to_parquet_incomplete(tmp_path):
    @delayed
    def create_dataframe_delayed(i):
        if i == 1:
            raise ValueError()
        return pd.DataFrame({"a": np.arange(2), "b": np.arange(2)})

    meta = dd.utils.make_meta(
        [
            ("a", np.int64),
            ("b", np.int64),
        ]
    )
    df = dd.from_delayed(
        [create_dataframe_delayed(0), create_dataframe_delayed(1)], meta=meta
    )
    path = os.path.join(tmp_path, "test.parquet")
    try:
        _dd_to_parquet(df, path, compute=True)
    except ValueError:
        assert os.path.exists(f"{path}.scallops")
        assert not is_parquet_file(path)


@pytest.mark.io
def test_to_parquet_complete(tmp_path):
    df = dd.from_pandas(pd.DataFrame({"a": np.arange(2), "b": np.arange(2)}))
    path = os.path.join(tmp_path, "test.parquet")
    _dd_to_parquet(df, path, compute=True)
    assert not os.path.exists(f".#{path}.scallops")
    assert is_parquet_file(path)


@pytest.mark.io
def test_to_parquet_no_compute(tmp_path):
    df = dd.from_pandas(pd.DataFrame({"a": np.arange(2), "b": np.arange(2)}))
    path = os.path.join(tmp_path, "test.parquet")
    result = _dd_to_parquet(df, path, compute=False)
    assert not os.path.exists(path) or len(os.listdir(path)) == 0
    assert os.path.exists(f"{path}.scallops")
    result.compute()
    assert not os.path.exists(f"{path}.scallops")
    assert is_parquet_file(path)


@pytest.mark.io
def test_to_parquet_remove_old_files(tmp_path):
    df = dd.from_pandas(pd.DataFrame({"a": np.arange(2), "b": np.arange(2)}))
    path = tmp_path / "test.parquet"
    path.mkdir()
    old_path = path / "foo.parquet"
    old_path.touch()
    assert old_path.exists()
    _dd_to_parquet(df, str(path), compute=True)
    assert not old_path.exists()


@pytest.mark.io
def test_match_images():
    img1 = xr.DataArray(
        np.random.random((1, 1, 1, 10, 10)), dims=["t", "c", "z", "y", "x"]
    )
    img2 = xr.DataArray(
        np.random.random((1, 1, 1, 11, 12)), dims=["t", "c", "z", "y", "x"]
    )
    images = [img1, img2]
    _match_size(images)
    for image in images:
        assert image.sizes["y"] == 11
        assert image.sizes["x"] == 12
    np.testing.assert_equal(img2.values, images[1].values)
    np.testing.assert_equal(img1.values, images[0].values[..., 1:, 2:])


@pytest.mark.io
@pytest.mark.parametrize("scenes", [True, False, ["Image:1"]])
def test_read_experiment_multi_scene(scenes):
    exp = read_experiment(
        f"{DATA_DIR}/tif",
        "{well}_3_t_1_c_3_z_5.ome.tiff",
        group_by=("well",),
        scenes=scenes,
    )
    if isinstance(scenes, list):
        assert list(exp.images.keys()) == ["s-Image:1"], list(exp.images.keys())
    elif scenes:
        assert list(exp.images.keys()) == ["s-Image:0", "s-Image:1", "s-Image:2"], list(
            exp.images.keys()
        )
        assert (
            exp.images["s-Image:0"].data != exp.images["s-Image:1"].data
        ).sum() == 2256811
    else:  # 1st scene
        assert list(exp.images.keys()) == ["s"], list(exp.images.keys())


@pytest.mark.io
@pytest.mark.parametrize("use_dask", [True, False])
def test_read_tif(use_dask):
    """Ensures that we can read a tif file using bioio.

    # In older versions of bioio the following was needed:
    # override default tif(f) reader to prevent the following exception:
    #   File "python3.10/site-packages/bioio/readers/ome_tiff_reader.py", line 317, in _read_immediate
    #     dims, coords = metadata_utils.get_dims_and_coords_from_ome(
    #   File "python3.10/site-packages/bioio/metadata/utils.py", line 632, in get_dims_and_coords_from_ome
    #     scene_meta = ome.images[scene_index]
    # bioio.formats.FORMAT_IMPLEMENTATIONS["tiff"] = ["bioio.readers.tiff_reader.TiffReader"]
    # bioio.formats.FORMAT_IMPLEMENTATIONS["tif"] = ["bioio.readers.tiff_reader.TiffReader"]
    """
    data = read_image(
        f"{DATA_DIR}/tif/10X_c0-DAPI-p65ab_A1_Tile-7.phenotype.tif",
        dask=use_dask,
    )
    if use_dask:
        data2 = read_image(
            f"{DATA_DIR}/tif/10X_c0-DAPI-p65ab_A1_Tile-7.phenotype.tif",
            dask=False,
        )
        np.testing.assert_array_equal(data.compute().values, data2.values)
    assert len(data.shape) == 5


@pytest.mark.parametrize("zarr_format", ["ome_zarr", "zarr"])
@pytest.mark.parametrize("use_dask", [True, False])
@pytest.mark.io
def test_write_zarr_attrs_coords_spacing(tmp_path, zarr_format, use_dask):
    data = xr.DataArray(
        np.arange(120, dtype=np.uint8).reshape((2, 3, 5, 4)),
        dims=["t", "c", "y", "x"],
        coords={"t": [3, 5], "c": ["foo1", "foo2", "foo3"]},
        attrs={
            "foo": "bar",
            "physical_pixel_sizes": (1.3, 1.2),
            "physical_pixel_units": ("angstrom", "mm"),
        },
    )
    if use_dask:
        data.data = da.from_array(data.data)
    zarr_path = str(tmp_path / "test.zarr")
    _write_zarr_image("foo", open_ome_zarr(zarr_path), data, zarr_format=zarr_format)
    img = read_image(f"{zarr_path}/images/foo", dask=False)
    assert img.attrs.pop("t") == [3, 5]
    xr.testing.assert_identical(data, img)
    assert get_image_spacing(img.attrs) == (1.3, 1.2)
    assert img.attrs["foo"] == "bar"
    assert img.attrs["physical_pixel_units"] == ("angstrom", "mm")
    assert img.coords["t"].values.tolist() == [3, 5]
    assert img.coords["c"].values.tolist() == ["foo1", "foo2", "foo3"]
    g = zarr.open(f"{zarr_path}/images/foo", mode="r")
    assert list(g.keys()) == ["s0"]
    attrs = _ome_attrs(g.attrs.asdict())
    version = _current_format().version
    assert {
        "channels": [
            {
                "label": "foo1",
                "color": "00FFFF",
                "window": {"min": 0, "max": 255, "start": 0, "end": 255},
            },
            {
                "label": "foo2",
                "color": "FFFF00",
                "window": {"min": 0, "max": 255, "start": 0, "end": 255},
            },
            {
                "label": "foo3",
                "color": "FF00FF",
                "window": {"min": 0, "max": 255, "start": 0, "end": 255},
            },
        ]
    } == attrs["omero"]
    expected_multiscale = {
        "datasets": [
            {
                "path": "s0",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 1.0, 1.3, 1.2]},
                    {"type": "translation", "translation": [0.0, 0.0, 0.0, 0.0]},
                ],
            }
        ],
        "name": "/images/foo",
        "metadata": {"foo": "bar", "t": [3, 5]},
        "axes": [
            {"name": "t", "type": "time"},
            {"name": "c", "type": "channel"},
            {"name": "y", "type": "space", "unit": "angstrom"},
            {"name": "x", "type": "space", "unit": "mm"},
        ],
    }
    if version == "0.4":
        # OME-Zarr 0.5 holds the version in "ome" instead
        expected_multiscale["version"] = version
    assert [expected_multiscale] == attrs["multiscales"]


@pytest.mark.io
def test_write_non_ome_zarr_image_no_dims(tmp_path):
    a = np.ones((3, 2))
    zarr_path = str(tmp_path / "test.zarr")
    _write_zarr_image("foo", open_ome_zarr(zarr_path), a, zarr_format="zarr")
    data_zarr = read_image(f"{zarr_path}/images/foo", dask=False)
    np.testing.assert_array_equal(data_zarr.values, a)


@pytest.mark.io
@pytest.mark.parametrize("use_dask", [True, False])
def test_write_non_ome_zarr_image(tmp_path, use_dask):
    image = read_image(
        f"{DATA_DIR}/tif/10X_c0-DAPI-p65ab_A1_Tile-7.phenotype.tif",
        dask=use_dask,
    )
    image.attrs = {"test": "1"}
    image.attrs["physical_pixel_sizes"] = (1, 1, 1)
    image.attrs["physical_pixel_units"] = ("millimeter", "millimeter", "millimeter")
    image.name = None
    zarr_path = str(tmp_path / "test1.zarr")
    _write_zarr_image("image1", open_ome_zarr(zarr_path), image, zarr_format="zarr")
    data_zarr = read_image(f"{zarr_path}/images/image1", dask=False)
    xr.testing.assert_identical(data_zarr, image)

    ome_zarr_path = str(tmp_path / "test2.zarr")
    _write_zarr_image("image1", open_ome_zarr(ome_zarr_path), image)
    data_ome_zarr = read_image(f"{ome_zarr_path}/images/image1", dask=False)
    xr.testing.assert_identical(data_ome_zarr, image)


@pytest.mark.io
def test_irregular_chunks(tmp_path):
    x1 = da.concatenate((da.ones((3, 10)), da.ones((2, 10)), da.ones((4, 10))), axis=0)
    anndata.AnnData(X=x1).write_zarr(tmp_path / "test.zarr")
    d = read_anndata(tmp_path / "test.zarr", dask=True)
    assert x1.chunks == d.X.chunks
    np.testing.assert_array_equal(x1.compute(), d.X.compute())


@pytest.mark.io
def test_experiment_file_list():
    image_paths = (
        f"{DATA_DIR}/experimentC/10X_c0-DAPI-p65ab/10X_c0-DAPI-p65ab_A1_Tile-102.phenotype.tif",
        f"{DATA_DIR}/experimentC/10X_c0-DAPI-p65ab/10X_c0-DAPI-p65ab_A1_Tile-",
    )

    gen = list(_set_up_experiment(image_paths, group_by=("*",)))
    assert len(gen) == 1
    x = gen[0]
    assert x[0] == ("image",)
    assert len(x[1]) == 2, "Should have 2 files"
    assert x[2]["id"] == "image", "Image id should be `image`"


@pytest.mark.io
def test_experiment_pattern_prefix(tmp_path):
    image_pattern = "c{t}/Well{well}_Point{point}_{tile}_Channel{channel}_Seq{seq}.tif"
    image_no_match = "xxx/Well{well}_Point{point}_{tile}_Channel{channel}_Seq{seq}.tif"
    src_path = f"{DATA_DIR}/experimentC/10X_c0-DAPI-p65ab/10X_c0-DAPI-p65ab_A1_Tile-102.phenotype.tif"
    times = ["1", "2"]
    for c in times:
        dest = tmp_path / image_pattern.format(
            t=c, well="A1", tile="001", point="001", channel=c, seq=c
        )
        (tmp_path / f"c{c}").mkdir(exist_ok=True)
        shutil.copy(src_path, dest)

    dest = tmp_path / image_no_match.format(
        well="A1", tile="001", point="001", channel="xxx", seq="xxx"
    )
    (tmp_path / "xxx").mkdir(exist_ok=True)
    shutil.copy(src_path, dest)
    exp = read_experiment(
        tmp_path, files_pattern=image_pattern, group_by=("well", "tile")
    )
    assert len(exp.images) == 1
    assert exp.images["A1-001"].sizes["t"] == 2


@pytest.mark.io
@pytest.mark.parametrize("use_dask", [True, False])
def test_experiment_separate_t_c(use_dask, tmp_path):
    """Test reading in exp where channels and cycles are both stored in separate images."""
    ncycles = 4
    nchannels = 5
    tmp_path = str(tmp_path)
    pattern = "{ignore}_cycle_{t}_yyy_{well}_{ignore}_w{c}.tif"

    for cycle in range(ncycles):
        for channel in range(nchannels):
            name = "xxx_cycle_{}_yyy_B06_s20_w{}.tif".format(cycle + 1, channel)
            path = os.path.join(tmp_path, name)
            value = cycle * 10 + channel + 1
            tmp = np.zeros((10, 10))
            tmp[:] = value
            save_ome_tiff(data=tmp, uri=path)

    # reading with dask fails with version bioio==4.10.0
    image = read_experiment(
        tmp_path, pattern, group_by=("well",), dask=use_dask
    ).images["B06"]
    np.testing.assert_equal(image.coords["t"].data, [1, 2, 3, 4])
    np.testing.assert_equal(image.coords["c"].data, ["0", "1", "2", "3", "4"])
    for cycle in range(ncycles):
        for channel in range(nchannels):
            test_image = np.zeros((10, 10))
            value = cycle * 10 + channel + 1
            test_image[:] = value
            np.testing.assert_equal(
                image.isel(c=channel, t=cycle).squeeze().data,
                test_image,
                f"dask: {use_dask}, channel: {channel}, cycle: {cycle}, value: {value}",
            )
    gen = list(_set_up_experiment(tmp_path, pattern, ("well",)))
    assert len(gen) == 1
    image_key, filepaths, metadata = gen[0]

    keys, filepaths, fileattrs = _group_src_attrs(
        metadata=metadata, metadata_fields=("c", "t")
    )
    assert len(filepaths) == 1

    assert len(filepaths[0]) == ncycles * nchannels


@pytest.mark.io
@pytest.mark.parametrize("use_dask", [True, False])
def test_group_by_one_field(use_dask):
    exp = read_experiment(
        f"{DATA_DIR}/experimentC/input",
        "10X_c{t}-SBS-{t}/{mag}X_c{t}-{exp}-{t}_{well}_Tile-102.{datatype}.tif",
        group_by=("well",),
        dask=use_dask,
    )
    assert len(exp.images) == 1
    image = exp.images["A1"]
    np.testing.assert_equal(image.coords["t"].data, [1, 2, 3, 4, 5, 7, 8, 9, 10])
    np.testing.assert_equal(
        image.coords["c"].data,
        ["Channel:0:0", "Channel:0:1", "Channel:0:2", "Channel:0:3", "Channel:0:4"],
    )


@pytest.mark.io
def test_experiment(experiment_c):
    exp = experiment_c
    assert len(exp.images) == 2
    image = exp.images["A1-102"]
    assert image.dims == ("t", "c", "z", "y", "x")
    np.testing.assert_equal(image.coords["t"].data, [1, 2, 3, 4, 5, 7, 8, 9, 10])
    np.testing.assert_equal(
        image.coords["c"].data,
        ["Channel:0:0", "Channel:0:1", "Channel:0:2", "Channel:0:3", "Channel:0:4"],
    )


@pytest.mark.io
def test_read_write_labels(tmp_path, experiment_c_A1_102_nuclei):
    nuclei = experiment_c_A1_102_nuclei.squeeze().data

    _write_zarr_labels(
        name="test", root=open_ome_zarr(str(tmp_path), "w"), labels=nuclei
    )
    test = read_ome_zarr_array(zarr.open(str(tmp_path / "labels" / "test"), mode="r"))
    np.testing.assert_equal(nuclei, test.data)


@pytest.mark.io
def test_read_write_experiment(experiment_c, tmp_path):
    path = os.path.join(tmp_path, "test.zarr")
    experiment_c.save(path)
    exp = read_experiment(path, files_pattern=None)

    assert len(exp.images) == 2

    assert exp.images["A1-102"].dims == ("t", "c", "z", "y", "x")

    np.testing.assert_equal(
        exp.images["A1-102"].coords["c"].data,
        ["Channel:0:0", "Channel:0:1", "Channel:0:2", "Channel:0:3", "Channel:0:4"],
        err_msg="c not equal",
    )
    np.testing.assert_equal(
        exp.images["A1-102"].coords["t"].data,
        [1, 2, 3, 4, 5, 7, 8, 9, 10],
        err_msg="t not equal",
    )

    np.testing.assert_equal(
        exp.images["A1-102"].values,
        experiment_c.images["A1-102"].values,
        err_msg="102 not equal",
    )
    np.testing.assert_equal(
        exp.images["A1-103"].values,
        experiment_c.images["A1-103"].values,
        err_msg="103 not equal",
    )

    assert exp.images["A1-102"].attrs["group"] == dict(tile="102", well="A1"), (
        "group not equal"
    )


@pytest.mark.io
def test_read_experiment_subset(experiment_c, tmp_path):
    path = os.path.join(tmp_path, "test.zarr")
    experiment_c.save(path)
    exp = read_experiment(path, files_pattern=None)
    assert len(exp.images) == 2, f"All images, {len(exp.images)}"
    exp = read_experiment(path, files_pattern=None, subset=("*-102",))
    assert len(exp.images) == 1, f"*-102, {len(exp.images)}"
    exp = read_experiment(path, files_pattern=None, subset=("A1-102",))
    assert len(exp.images) == 1, f"A1-102, {len(exp.images)}"
    exp = read_experiment(path, files_pattern=None, subset=("A1-103",))
    assert len(exp.images) == 1, f"A1-103, {len(exp.images)}"
    exp = read_experiment(path, files_pattern=None, subset=("A1-*",))
    assert len(exp.images) == 2, f"A1-*, {len(exp.images)}"
    exp = read_experiment(path, files_pattern=None, subset=("A1",))
    assert len(exp.images) == 0, f"A1, {len(exp.images)}"


@pytest.mark.io
def test_set_up_experiment_zarr(experiment_c, tmp_path):
    path = os.path.join(tmp_path, "test.zarr")
    experiment_c.save(path)
    for group, file_list, metadata in _set_up_experiment(path):
        assert group in [("A1-102",), ("A1-103",)]
        assert metadata["id"] in ["A1-102", "A1-103"]
        img = _images2fov(file_list, metadata)
        assert img.shape == (9, 5, 1, 1024, 1024)


@pytest.mark.io
def test_set_up_experiment_zarr_group_stack(tmp_path):
    path = os.path.join(tmp_path, "test.zarr")
    experiment = Experiment()
    image = np.array([skimage.data.camera()])
    experiment.images["A1-1"] = xr.DataArray(
        image, dims=["t", "y", "x"], coords={"t": [1]}
    )
    experiment.images["A1-2"] = xr.DataArray(
        image, dims=["t", "y", "x"], coords={"t": [2]}
    )
    experiment.save(path)
    gen = list(_set_up_experiment(path, files_pattern="{well}-{t}", group_by=("well",)))
    assert len(gen) == 1
    item = gen[0]
    assert item[0] == ("A1",)
    assert item[2]["id"] == "A1"
    img = _images2fov(item[1], item[2])
    assert img.shape == (2, 512, 512)


@pytest.mark.io
def test_set_up_experiment_no_groupby():
    gen = list(
        _set_up_experiment(
            f"{DATA_DIR}/experimentC/input",
            files_pattern="10X_c{t}-SBS-{t}/{mag}X_c{t}-{exp}-{t}_{well}_Tile-{tile}.{datatype}.tif",
        )
    )
    assert len(gen) == 18  # 9 cycles and 2 tiles


@pytest.mark.io
@pytest.mark.parametrize("channel_dim", [True, False])
def test_to_image_montage(tmp_path, channel_dim):
    nimages = 3
    experiment = Experiment()
    tile_size = (2, 4, 3) if channel_dim else (4, 3)

    for i in range(nimages):
        array = np.zeros(tile_size)
        array[:] = i + 1
        experiment.images[str(i + 1)] = xr.DataArray(
            array, dims=["c", "y", "x"] if channel_dim else ["y", "x"]
        )
    path = os.path.join(tmp_path, "test.tif")
    keys = list(experiment.images.keys())
    to_image_montage(experiment, keys, path)
    image = read_image(path).squeeze().values
    assert len(image.shape) == 3 if channel_dim else len(image.shape) == 2
    assert image.shape[-2] == 8
    assert image.shape[-1] == 6
    # output is 2x2 grid (montage is always square)
    # last tile is empty because we input 3 images

    grid_size = 2
    for i in range(grid_size):
        for j in range(grid_size):
            if channel_dim:
                array = image[
                    :,
                    i * tile_size[-2] : i * tile_size[-2] + tile_size[-2],
                    j * tile_size[-1] : j * tile_size[-1] + tile_size[-1],
                ]
            else:
                array = image[
                    i * tile_size[-2] : i * tile_size[-2] + tile_size[-2],
                    j * tile_size[-1] : j * tile_size[-1] + tile_size[-1],
                ]
            index = i * grid_size + j
            value = index + 1 if index < len(keys) else 0
            assert np.all(array == value)


@pytest.mark.io
def test_read_write_experiment_nd2(tmp_path):
    # tests serializing nd2 attrs to json
    # from https://cellpainting-gallery.s3.amazonaws.com/cpg0021-periscope/broad/images/20200805_A549_WG_Screen/images/CP186A/10X_c10-SBS-10/Well1_Point1_0001_ChannelDAPI%2CCy3%2CA594%2CCy5%2CCy7_Seq0001.nd2
    file_pattern = "Well{well}_Point{skip}_{skip}_Seq{tile}.nd2"
    exp = read_experiment(os.path.join(DATA_DIR, "nd2"), file_pattern)
    path = os.path.join(tmp_path, "test.zarr")
    exp.save(path)
    exp = read_experiment(path)
    assert len(exp.images) == 1


@pytest.mark.io
def test_read_write_image_spacing_stack(tmp_path):
    image1 = read_image(
        f"{DATA_DIR}/experimentC/10X_c0-DAPI-p65ab/10X_c0-DAPI-p65ab_A1_Tile-102.phenotype.tif"
    ).squeeze()
    image2 = read_image(
        f"{DATA_DIR}/experimentC/10X_c0-DAPI-p65ab/10X_c0-DAPI-p65ab_A1_Tile-103.phenotype.tif"
    ).squeeze()
    img_path = os.path.join(tmp_path, "images")
    os.makedirs(img_path, exist_ok=True)
    images = [image1, image2]
    for i in range(len(images)):
        img = images[i]
        path = os.path.join(img_path, "A1_{}.tiff".format(i))
        save_ome_tiff(img.values, uri=path)
        img = read_image(path)
        if isinstance(img.attrs["processed"], str):
            img.attrs["processed"] = ome_types.from_xml(img.attrs["processed"])
        img.attrs["processed"].images[0].pixels.physical_size_x = 1
        img.attrs["processed"].images[0].pixels.physical_size_y = 1
        save_ome_tiff(img.values, uri=path, ome_xml=img.attrs["processed"].to_xml())
    exp = read_experiment(img_path, "{well}_{tile}.tiff", group_by=("well", "tile"))
    exp.save(os.path.join(img_path, "images.zarr"))
    exp2 = read_experiment(os.path.join(img_path, "images.zarr"))
    image = exp2.images["A1-0"]
    assert get_image_spacing(image.attrs) == (1.0, 1.0)


@pytest.mark.io
def test_read_file_sort_order(tmp_path):
    img = read_image(
        f"{DATA_DIR}/experimentC/10X_c0-DAPI-p65ab/10X_c0-DAPI-p65ab_A1_Tile-102.phenotype.tif"
    ).squeeze()
    img_path = os.path.join(tmp_path, "images")
    os.makedirs(img_path, exist_ok=True)
    save_ome_tiff(
        img.values, uri=os.path.join(img_path, "A1-20231010_20x_6W_FISH.tiff")
    )
    save_ome_tiff(img.values, uri=os.path.join(img_path, "A1-20231012_20x_6W_IF.tiff"))
    image_gen = list(
        _set_up_experiment(
            img_path,
            "{well}-{t}.tiff",
            ("well",),
        )
    )
    assert len(image_gen) == 1
    image_gen = image_gen[0]
    _, file_list, metadata = image_gen
    assert len(file_list) == 2
    assert os.path.basename(file_list[0]) == "A1-20231010_20x_6W_FISH.tiff"
    assert os.path.basename(file_list[1]) == "A1-20231012_20x_6W_IF.tiff"
    image_gen = list(
        _set_up_experiment(
            img_path,
            "{well}-{t}.tiff",
            ("well",),
            file_sort_order={"t": ["20231012_20x_6W_IF", "20231010_20x_6W_FISH"]},
        )
    )[0]

    _, file_list, metadata = image_gen
    assert os.path.basename(file_list[0]) == "A1-20231012_20x_6W_IF.tiff"
    assert os.path.basename(file_list[1]) == "A1-20231010_20x_6W_FISH.tiff"


@pytest.mark.io
def test_read_file_sort_order_numeric_values(tmp_path):
    # listed values must sort before unlisted values, even numeric ones such as "0"
    img_path = os.path.join(tmp_path, "images")
    os.makedirs(img_path, exist_ok=True)
    for t in ["0", "1", "10", "2", "IF"]:
        save_ome_tiff(
            np.zeros((4, 4), dtype=np.uint16),
            uri=os.path.join(img_path, f"A1-{t}.tiff"),
        )
    _, file_list, _ = list(
        _set_up_experiment(
            img_path,
            "{well}-{t}.tiff",
            ("well",),
            file_sort_order={"t": ["IF"]},
        )
    )[0]
    assert [os.path.basename(f) for f in file_list] == [
        "A1-IF.tiff",
        "A1-0.tiff",
        "A1-1.tiff",
        "A1-2.tiff",
        "A1-10.tiff",
    ]


@pytest.mark.io
def test_read_experiment_csv(tmp_path):
    exp = read_experiment(
        f"{DATA_DIR}/experimentC/input",
        "10X_c{t}-SBS-{t}/{ignore}X_c{t}-{ignore}-{t}_{well}_Tile-{tile}.{ignore}.tif",
        group_by=("well", "tile"),
    )
    paths = glob.glob(f"{DATA_DIR}/experimentC/input/**/*.tif")
    wells = []
    tiles = []
    times = []
    for path in paths:
        tokens = os.path.basename(path).split(
            "_"
        )  # e.g. : '10X_c5-SBS-5_A1_Tile-103.sbs.tif'
        times.append(tokens[1].split("-")[0][1:])
        wells.append(tokens[2])
        tiles.append("103" if tokens[3].startswith("Tile-103") else "102")
    csv_path = os.path.join(tmp_path, "test.csv")
    pd.DataFrame(dict(image=paths, well=wells, tile=tiles, t=times)).to_csv(
        csv_path, index=False
    )
    assert list(exp.images.keys()) == ["A1-102", "A1-103"]

    exp2 = read_experiment(
        csv_path,
        files_pattern=None,
        group_by=("well", "tile"),
    )
    assert list(exp2.images.keys()) == ["A1-102", "A1-103"]
    xr.testing.assert_equal(exp.images["A1-102"], exp2.images["A1-102"])
    xr.testing.assert_equal(exp.images["A1-103"], exp2.images["A1-103"])


@pytest.mark.io
def test_read_experiment_csv_different_directories(tmp_path):
    paths = []
    tiles = []
    channels = []
    for i, directory in enumerate(["zdir", "adir"]):
        os.makedirs(tmp_path / directory)
        for j, tile in enumerate(["1", "2"]):
            path = str(tmp_path / directory / f"e100_1_t0_A1_s{tile}_w{i}.tif")
            save_ome_tiff(np.full((4, 4), 10 * j + i, dtype=np.uint16), uri=path)
            paths.append(path)
            tiles.append(tile)
            channels.append(i)
    csv_path = os.path.join(tmp_path, "test.csv")
    pd.DataFrame(dict(image=paths, well="A1", tile=tiles, c=channels)).to_csv(
        csv_path, index=False
    )

    gen = list(
        _set_up_experiment(csv_path, files_pattern=None, group_by=("well", "tile"))
    )
    assert [g[0] for g in gen] == [("A1", "1"), ("A1", "2")]
    for j, (_, file_list, metadata) in enumerate(gen):
        image = _images2fov(file_list, metadata)
        np.testing.assert_array_equal(image.coords["c"].values, ["0", "1"])
        np.testing.assert_array_equal(
            image.isel(t=0, z=0).max(dim=("y", "x")), [10 * j, 10 * j + 1]
        )


@pytest.mark.io
def test_read_experiment_csv_group_by_str(tmp_path):
    path = str(tmp_path / "image.tif")
    save_ome_tiff(np.zeros((4, 4), dtype=np.uint16), uri=path)
    csv_path = os.path.join(tmp_path, "test.csv")
    pd.DataFrame(dict(image=[path], well=["A01"], tile=["001"])).to_csv(
        csv_path, index=False
    )
    exp = read_experiment(csv_path, group_by=("well", "tile"))
    assert list(exp.images.keys()) == ["A01-001"]


@pytest.mark.io
def test_dask_zarr_io_component(tmp_path, recwarn):
    # this is what current version of ome_zarr does
    path = tmp_path / "test.zarr"
    da.to_zarr(
        arr=da.random.random((2, 2)),
        url=path,
        component="0",
        fill_value=10,
    )
    assert zarr.open(path, mode="r")["0"].fill_value == 10
    assert len(recwarn) == 0, (
        f"Expected no warnings, but got: {[str(w.message) for w in recwarn]}"
    )


@pytest.mark.io
def test_dask_zarr_io(tmp_path, recwarn):
    # this is what current version of ome_zarr does
    path = tmp_path / "test.zarr"
    da.to_zarr(
        arr=da.random.random((10, 10), chunks=(5, 5)),
        url=path,
        compute=True,
        fill_value=10,
    )
    assert zarr.open(path, mode="r").fill_value == 10
    assert len(recwarn) == 0, (
        f"Expected no warnings, but got: {[str(w.message) for w in recwarn]}"
    )


@pytest.mark.io
def test_anndata_zarr(tmp_path):
    path = tmp_path / "test.zarr"
    d = anndata.AnnData(
        X=np.random.random((4, 4)),
        uns={"a": 1},
        var=pd.DataFrame({"a": [4, 3, 2, 1]}),
        obs=pd.DataFrame({"b": [1, 2, 3, 4]}),
    )
    write_anndata_zarr(d, path)
    d2 = read_anndata(path, dask=True)
    np.testing.assert_equal(d2.X.compute(), d.X)
    pd.testing.assert_frame_equal(d.obs, d2.obs)
    pd.testing.assert_frame_equal(d.var, d2.var)
    assert d.uns == d2.uns


@pytest.mark.io
def test_require_group_loses_create_race(tmp_path, monkeypatch):
    """A writer that loses the create race opens the winner's group."""
    root = open_ome_zarr(tmp_path / "test.zarr", mode="a")
    root.require_group("labels").create_group("A01")  # the winning well

    original = AsyncGroup.getitem
    missed = []

    async def miss_once(self, key):
        # Model the race: our lookup runs before the winner's write lands, so the
        # create that follows it collides.
        if not missed:
            missed.append(key)
            raise KeyError(key)
        return await original(self, key)

    monkeypatch.setattr(AsyncGroup, "getitem", miss_once)
    group = _require_group(root, "labels")
    assert missed == ["labels"]
    assert list(group.keys()) == ["A01"]


@pytest.mark.io
def test_require_group_array_conflict_still_raises(tmp_path):
    """An array at the requested path is a real error, not a lost race."""
    root = open_ome_zarr(tmp_path / "test.zarr", mode="a")
    root.create_array("labels", shape=(4,), dtype="i4")
    with pytest.raises(TypeError):
        _require_group(root, "labels")


@pytest.mark.io
def test_open_ome_zarr_loses_root_create_race(tmp_path, monkeypatch):
    """open_ome_zarr(mode="a") tolerates another writer creating the root group."""
    url = tmp_path / "test.zarr"
    open_ome_zarr(url, mode="a").require_group("labels")  # the winning well

    original = AsyncGroup.open
    missed = []

    @classmethod
    async def miss_once(cls, store_path, **kwargs):
        if not missed:
            missed.append(store_path)
            raise FileNotFoundError(store_path)
        return await original.__func__(cls, store_path, **kwargs)

    monkeypatch.setattr(AsyncGroup, "open", miss_once)
    root = open_ome_zarr(url, mode="a")
    assert len(missed) == 1
    assert list(root.keys()) == ["labels"]


@pytest.mark.io
def test_open_ome_zarr_loses_root_create_race_to_third_writer(tmp_path, monkeypatch):
    """The retry must fall back to a plain read, or a third racer can beat it too.

    A retry that re-attempts the same check-then-create ``zarr.open`` call can lose
    to *another* concurrent creator just as easily as the first attempt did. The
    fallback must instead do a "must exist" open (mode="r+"), which never attempts
    to create and so cannot lose a create race.
    """
    url = tmp_path / "test.zarr"
    open_ome_zarr(url, mode="a").require_group("labels")  # the winning well

    original_open = zarr.open

    def open_that_always_loses_create(store, *, mode, zarr_format):
        if mode == "a":
            raise ContainsGroupError("root group exists")
        return original_open(store, mode=mode, zarr_format=zarr_format)

    monkeypatch.setattr(zarr_io.zarr, "open", open_that_always_loses_create)
    root = open_ome_zarr(url, mode="a")
    assert list(root.keys()) == ["labels"]


def _assert_anndata_equal(adata1: anndata.AnnData, adata2: anndata.AnnData):
    np.testing.assert_array_equal(adata1.X, adata2.X)
    pd.testing.assert_frame_equal(adata1.obs, adata2.obs)
    pd.testing.assert_frame_equal(adata1.var, adata2.var)
    assert adata1.layers.keys() == adata2.layers.keys()
    assert adata1.obsm.keys() == adata2.obsm.keys()
    assert adata1.varm.keys() == adata2.varm.keys()
    for key in adata1.layers.keys():
        np.testing.assert_array_equal(
            adata1.layers[key], adata2.layers[key], err_msg=f"Layer {key}"
        )

    for key in adata1.obsm.keys():
        np.testing.assert_array_equal(
            adata1.obsm[key], adata2.obsm[key], err_msg=f"obsm {key}"
        )
    for key in adata1.varm.keys():
        np.testing.assert_array_equal(
            adata1.varm[key], adata2.varm[key], err_msg=f"varm {key}"
        )


@pytest.mark.io
def test_slice_anndata():
    d = anndata.AnnData(
        X=np.arange(4).reshape((2, 2)),
        obs=pd.DataFrame(index=["1", "2"]),
        var=pd.DataFrame(index=["1", "2"]),
        layers={"test": np.arange(4).reshape((2, 2))},
        obsm={"test": np.arange(4).reshape((2, 2))},
        varm={"test": np.arange(4).reshape((2, 2))},
    )
    data1 = d[[1, 0], [1, 0]]
    data2 = _slice_anndata(d, [1, 0], [1, 0])
    _assert_anndata_equal(data1, data2)

    data1 = d[d.obs.index.isin(["2"]), d.var.index.isin(["1"])]
    data2 = _slice_anndata(d, d.obs.index.isin(["2"]), d.var.index.isin(["1"]))
    _assert_anndata_equal(data1, data2)


def _save_labels(path: str, labels: dict[str, np.ndarray]) -> zarr.Group:
    exp = Experiment()
    for name, value in labels.items():
        exp.labels[name] = xr.DataArray(value, dims=["y", "x"])
    exp.save(path)
    return zarr.open(path, mode="r")["labels"]


@pytest.mark.io
def test_label_args_type(tmp_path):
    path = str(tmp_path / "test.zarr")
    _save_labels(path, {"A1-cell": np.ones((4, 4), dtype=np.uint32)})
    assert "A1-cell" in _label_args_type(path)

    with pytest.raises(argparse.ArgumentTypeError, match="Unable to open"):
        _label_args_type(str(tmp_path / "missing.zarr"))

    no_labels_path = str(tmp_path / "no-labels.zarr")
    zarr.open_group(no_labels_path, mode="w").create_group("images")
    with pytest.raises(argparse.ArgumentTypeError, match="Labels group not found"):
        _label_args_type(no_labels_path)


@pytest.mark.io
def test_find_label_group_last_wins(tmp_path):
    group1 = _save_labels(
        str(tmp_path / "1.zarr"),
        {
            "A1-cell": np.full((4, 4), 1, dtype=np.uint32),
            "A1-nuclei": np.full((4, 4), 3, dtype=np.uint32),
        },
    )
    group2 = _save_labels(
        str(tmp_path / "2.zarr"), {"A1-cell": np.full((4, 4), 2, dtype=np.uint32)}
    )

    def _value(label_groups, key):
        return np.unique(read_ome_zarr_array(find_label_group(label_groups, key)))

    assert _value([group1, group2], "A1-cell") == [2]
    assert _value([group2, group1], "A1-cell") == [1]
    # only in the first store
    assert _value([group1, group2], "A1-nuclei") == [3]
    with pytest.raises(ValueError, match="A1-cytosol"):
        find_label_group([group1, group2], "A1-cytosol")


@pytest.mark.io
def test_last_label_match():
    assert last_label_match([("a", {}), ("b", {})]) == [("b", {})]
    assert last_label_match([("a", {})]) == [("a", {})]


@pytest.mark.io
def test_set_up_experiment_filter_matches(tmp_path):
    group1 = _save_labels(
        str(tmp_path / "1.zarr"), {"A1-cell": np.full((4, 4), 1, dtype=np.uint32)}
    )
    group2 = _save_labels(
        str(tmp_path / "2.zarr"), {"A1-cell": np.full((4, 4), 2, dtype=np.uint32)}
    )
    kwargs = dict(
        image_path=[group1, group2],
        files_pattern="{well}-{label_suffix}",
        group_by=["well", "label_suffix"],
    )
    gen = list(_set_up_experiment(**kwargs))
    assert len(gen) == 1
    assert len(gen[0][1]) == 2  # duplicates from both stores without a filter

    gen = list(_set_up_experiment(**kwargs, filter_matches=last_label_match))
    assert len(gen) == 1
    group, file_list, _ = gen[0]
    assert group == ("A1", "cell")
    assert len(file_list) == 1
    assert np.unique(read_ome_zarr_array(file_list[0])) == [2]


@pytest.mark.parametrize("fmt", [FormatV04(), FormatV05()], ids=lambda f: f.version)
def test_zarr_attrs_round_trip(fmt):
    from types import SimpleNamespace

    from scallops.stitch.utils import _pixel_size_from_image, get_tile_position

    axes = [{"name": "c", "type": "channel"}, {"name": "y"}, {"name": "x"}]
    image_attrs = {
        "omero": {"channels": [{"label": "DAPI"}, {"label": "FITC"}]},
        "position_y": 10.0,
        "position_x": 20.0,
        "position_y_unit": "µm",
        "position_x_unit": "µm",
        "physical_size_y": 0.5,
        "physical_size_x": 0.25,
        "physical_size_y_unit": "µm",
        "physical_size_x_unit": "µm",
    }
    grp = zarr.open_group(
        zarr.storage.MemoryStore(), mode="w", zarr_format=fmt.zarr_format
    )
    zarr_attrs = zarr_io._create_zarr_attrs(
        fmt, SimpleNamespace(name="A-3"), ["c", "y", "x"], image_attrs, axes, None
    )
    assert "omero" in image_attrs  # input is not mutated
    _add_metadata(grp, zarr_attrs, fmt)
    attrs = grp.attrs.asdict()
    if fmt.version == "0.5":
        assert set(attrs) == {"ome"}
        assert attrs["ome"]["version"] == "0.5"
        assert "version" not in attrs["ome"]["multiscales"][0]
    else:
        assert attrs["multiscales"][0]["version"] == "0.4"

    coords, metadata, dims = zarr_io._read_zarr_attrs(attrs)
    assert dims == ["c", "y", "x"]
    assert coords["c"] == ["DAPI", "FITC"]
    assert metadata["position_y"] == 10.0
    assert metadata["position_x"] == 20.0

    image = SimpleNamespace(
        ome_metadata=None, metadata=SimpleNamespace(attributes=attrs)
    )
    np.testing.assert_array_equal(get_tile_position(image), [10.0, 20.0])
    np.testing.assert_array_equal(_pixel_size_from_image(image), [0.5, 0.25])
    assert zarr_io._has_multiscales(attrs)


def test_read_ome_zarr_root_without_multiscales(monkeypatch):
    # an OME-Zarr 0.5 root holds only a version; its image is under images/
    monkeypatch.setattr(zarr_io, "_current_format", FormatV05)
    root = zarr.open_group(zarr.storage.MemoryStore(), mode="w", zarr_format=3)
    root.attrs["ome"] = {"version": "0.5"}
    data = np.arange(12, dtype=np.uint16).reshape(3, 4)
    zarr_io._write_zarr_image("A-3", root, data)
    assert not zarr_io.is_ome_zarr_array(root)
    assert zarr_io.is_ome_zarr_array(root["images/A-3"])
    np.testing.assert_array_equal(zarr_io.read_ome_zarr_array(root).values, data)


@pytest.mark.parametrize("fmt", [FormatV04(), FormatV05()], ids=lambda f: f.version)
def test_physical_pixel_size_from_scale(fmt):
    # scallops' writer stores the pixel size in the scale transformation and axis
    # units, not in multiscales metadata
    from types import SimpleNamespace

    from scallops.stitch.utils import _pixel_size_from_image

    image_attrs = {
        "physical_pixel_sizes": (0.5, 0.25),
        "physical_pixel_units": ("micrometer", "nanometer"),
    }
    image_attrs, axes, scale_dict = zarr_io._attrs_axes_scales(
        image_attrs, None, ["c", "y", "x"], np.dtype("uint16")
    )
    attrs = zarr_io._create_zarr_attrs(
        fmt, SimpleNamespace(name="A-3"), ["c", "y", "x"], image_attrs, axes, scale_dict
    )
    assert zarr_io._physical_pixel_size(attrs) == (
        [0.5, 0.25],
        "micrometer",
        "nanometer",
    )
    assert zarr_io._dataset_path(attrs) == "s0"
    image = SimpleNamespace(
        ome_metadata=None, metadata=SimpleNamespace(attributes=attrs)
    )
    np.testing.assert_allclose(_pixel_size_from_image(image), [0.5, 0.25e-3])


def test_physical_pixel_size_without_units():
    attrs = {
        "multiscales": [
            {
                "axes": [{"name": "y"}, {"name": "x"}],
                "datasets": [
                    {
                        "path": "s0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 1.0]}
                        ],
                    }
                ],
            }
        ]
    }
    assert zarr_io._physical_pixel_size(attrs) is None
    assert zarr_io._dataset_path(attrs) == "s0"
    assert zarr_io._dataset_path({}) == "0"


def test_add_metadata_merges_ome():
    grp = zarr.open_group(zarr.storage.MemoryStore(), mode="w", zarr_format=3)
    grp.attrs.update({"ome": {"version": "0.5", "labels": ["A-3-cell"]}, "other": 1})
    _add_metadata(grp, {"multiscales": [{"name": "A-3"}]}, FormatV05())
    assert grp.attrs["ome"] == {
        "version": "0.5",
        "labels": ["A-3-cell"],
        "multiscales": [{"name": "A-3"}],
    }
    assert grp.attrs["other"] == 1


@pytest.mark.parametrize("fmt", [FormatV04, FormatV05], ids=["0.4", "0.5"])
def test_write_labels_image_label_nesting(monkeypatch, fmt):
    # OME-Zarr 0.5 nests image-label under "ome" alongside multiscales
    monkeypatch.setattr(zarr_io, "_current_format", fmt)
    root = zarr.open_group(
        zarr.storage.MemoryStore(),
        mode="w",
        zarr_format=3 if fmt().version == "0.5" else 2,
    )
    data = np.arange(12, dtype=np.uint32).reshape(3, 4)
    image_label = {"source": {"image": "../../images/A-3"}}
    zarr_io._write_zarr_labels(
        "A-3", root, data, group_metadata={"image-label": image_label}
    )
    attrs = root["labels/A-3"].attrs.asdict()
    if fmt().version == "0.5":
        assert "image-label" not in attrs and "multiscales" not in attrs
        attrs = attrs["ome"]
        assert attrs["version"] == "0.5"
    assert attrs["image-label"] == image_label
    assert "multiscales" in attrs
    np.testing.assert_array_equal(
        zarr_io.read_ome_zarr_array(root["labels/A-3"]).values, data
    )


@pytest.mark.parametrize("fmt", [FormatV04(), FormatV05()], ids=lambda f: f.version)
def test_create_zarr_attrs_image_label_and_empty_omero(fmt):
    from types import SimpleNamespace

    axes = [{"name": "y"}, {"name": "x"}]
    attrs = zarr_io._create_zarr_attrs(
        fmt, SimpleNamespace(name="A-3"), ["y", "x"], {}, axes, None, image_label={}
    )
    assert attrs["image-label"] == {}
    assert "omero" not in attrs  # no channel metadata, so no empty omero


@pytest.mark.parametrize("fmt", [FormatV04, FormatV05], ids=["0.4", "0.5"])
def test_write_image_group_metadata_not_double_nested(monkeypatch, fmt):
    monkeypatch.setattr(zarr_io, "_current_format", fmt)
    version = fmt().version
    root = zarr.open_group(
        zarr.storage.MemoryStore(), mode="w", zarr_format=3 if version == "0.5" else 2
    )
    data = np.zeros((3, 4), dtype=np.uint8)
    zarr_io._write_zarr_image(
        "a", root, data, group_metadata={"ome": {"version": version, "foo": 1}}
    )
    attrs = root["images/a"].attrs.asdict()
    assert attrs["ome"]["foo"] == 1 and "ome" not in attrs["ome"]

    zarr_io._write_zarr_image(
        "b",
        root,
        np.zeros((2, 3, 4), dtype=np.uint8),
        zarr_format="zarr",
        group_metadata={"foo": 1},
    )
    attrs = root["images/b"].attrs.asdict()
    assert attrs["foo"] == 1
    assert "version" not in attrs.get("ome", {})


def test_pixel_size_parsers_agree_with_translation_first():
    # both parsers use the first transformation of type "scale"
    attrs = {
        "multiscales": [
            {
                "axes": [
                    {"name": "y", "unit": "micrometer"},
                    {"name": "x", "unit": "micrometer"},
                ],
                "datasets": [
                    {
                        "path": "s0",
                        "coordinateTransformations": [
                            {"type": "translation", "translation": [5.0, 6.0]},
                            {"type": "scale", "scale": [0.5, 0.25]},
                        ],
                    }
                ],
            }
        ]
    }
    _, metadata, _ = zarr_io._read_zarr_attrs(attrs)
    assert metadata["physical_pixel_sizes"] == (0.5, 0.25)
    assert zarr_io._physical_pixel_size(attrs) == (
        [0.5, 0.25],
        "micrometer",
        "micrometer",
    )


@pytest.mark.io
@pytest.mark.parametrize("fmt", [FormatV04(), FormatV05()], ids=lambda f: f.version)
def test_add_metadata_layout(fmt):
    attrs = {"image-label": {}, "custom": 1}
    grp = zarr.open_group(
        zarr.storage.MemoryStore(), mode="w", zarr_format=fmt.zarr_format
    )
    _add_metadata(grp, attrs, fmt)
    if fmt.version == "0.5":
        assert grp.attrs.asdict() == {
            "custom": 1,
            "ome": {"version": "0.5", "image-label": {}},
        }
        # an "ome" entry in attrs is merged, not nested
        _add_metadata(grp, {"ome": {"multiscales": []}, "omero": {}}, fmt)
        assert grp.attrs["ome"] == {
            "version": "0.5",
            "image-label": {},
            "multiscales": [],
            "omero": {},
        }
    else:
        assert grp.attrs.asdict() == attrs


@pytest.mark.io
def test_ome_attrs_unwraps_any_ome_entry():
    # a 0.5 labels group holds only "labels" under "ome"
    attrs = {"ome": {"version": "0.5", "labels": ["A-3"]}, "other": 1}
    assert zarr_io._ome_attrs(attrs)["labels"] == ["A-3"]
    # top-level keys are kept, keys under "ome" take precedence
    attrs = {"labels": ["old"], "omero": {}, "ome": {"labels": ["new"]}}
    ome = zarr_io._ome_attrs(attrs)
    assert ome["labels"] == ["new"] and "omero" in ome
    assert zarr_io._ome_attrs({"labels": ["A-3"]}) == {"labels": ["A-3"]}


@pytest.mark.io
def test_physical_pixel_size_prefers_metadata():
    multiscale0 = {
        "axes": [
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ],
        "datasets": [
            {
                "path": "s0",
                "coordinateTransformations": [{"type": "scale", "scale": [1.3, 1.3]}],
            }
        ],
        "metadata": {
            "physical_size_y": 0.65,
            "physical_size_x": 0.65,
            "physical_size_y_unit": "micrometer",
            "physical_size_x_unit": "micrometer",
        },
    }
    attrs = {"multiscales": [multiscale0]}
    assert _physical_pixel_size(attrs) == ([0.65, 0.65], "micrometer", "micrometer")
    # falls back to the scale when the metadata has no pixel size
    multiscale0["metadata"] = {}
    assert _physical_pixel_size(attrs) == ([1.3, 1.3], "micrometer", "micrometer")


@pytest.mark.parametrize("fmt", [FormatV04, FormatV05], ids=["0.4", "0.5"])
def test_scallops_created_arrays_validate(monkeypatch, tmp_path, fmt):
    # arrays created by scallops (not write_multiscale) need dimension_names for 0.5
    import importlib

    pytest.importorskip("ome_zarr_models")
    from scallops.stitch import fuse as fuse_mod
    from scallops.tests.test_stitch import _write_fuse_tiles

    monkeypatch.setattr(zarr_io, "_current_format", fmt)
    monkeypatch.setattr(fuse_mod, "_current_format", fmt)
    version = fmt().version
    models = importlib.import_module(f"ome_zarr_models.v0{version[-1]}")
    root = zarr_io.open_ome_zarr(str(tmp_path / "a.zarr"), mode="w")
    df, _, _ = _write_fuse_tiles(tmp_path)
    group = _require_group(root, "images").create_group("fused")
    fuse_mod._fuse(df, group=group, blend="linear", z_index=0, chunk_size=None)
    models.Image.from_zarr(group)
    assert zarr_io.read_ome_zarr_array(group).dims == ("c", "y", "x")
    if version == "0.5":
        assert list(group["s0"].metadata.dimension_names) == ["c", "y", "x"]
