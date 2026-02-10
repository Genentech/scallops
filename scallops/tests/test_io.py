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

from scallops.cli.util import _group_src_attrs
from scallops.experiment.elements import Experiment
from scallops.io import (
    _images2fov,
    _match_size,
    _set_up_experiment,
    _to_parquet,
    get_image_spacing,
    is_parquet_file,
    is_scallops_zarr,
    read_experiment,
    read_image,
    save_ome_tiff,
    to_image_montage,
)
from scallops.zarr_io import (
    _write_zarr_image,
    _write_zarr_labels,
    is_anndata_zarr,
    open_ome_zarr,
    read_ome_zarr_array,
)


@pytest.fixture(params=[False, True])
def dask(request):
    return request.param


@pytest.mark.io
def test_is_scallops_zarr(tmp_path):
    data = anndata.AnnData(
        X=np.ones((2, 2)),
        obs=pd.DataFrame(index=["1", "2"]),
        var=pd.DataFrame(index=["1", "2"]),
    )
    path = os.path.join(tmp_path, "test.zarr")
    anndata.io.write_zarr(path, data, convert_strings_to_categoricals=False)

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
    d.write_zarr(path1, convert_strings_to_categoricals=False)
    assert is_anndata_zarr(path1)

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
        d.write_zarr(path2, convert_strings_to_categoricals=False)
    except ValueError:
        pass
    assert not is_anndata_zarr(path2)


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
        _to_parquet(df, path, compute=True)
    except ValueError:
        assert os.path.exists(f"{path}.scallops")
        assert not is_parquet_file(path)


@pytest.mark.io
def test_to_parquet_complete(tmp_path):
    df = dd.from_pandas(pd.DataFrame({"a": np.arange(2), "b": np.arange(2)}))
    path = os.path.join(tmp_path, "test.parquet")
    _to_parquet(df, path, compute=True)
    assert not os.path.exists(f".#{path}.scallops")
    assert is_parquet_file(path)


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
        "scallops/tests/data/tif",
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
def test_read_tif(dask):
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
        "scallops/tests/data/tif/10X_c0-DAPI-p65ab_A1_Tile-7.phenotype.tif", dask=dask
    )
    if dask:
        data2 = read_image(
            "scallops/tests/data/tif/10X_c0-DAPI-p65ab_A1_Tile-7.phenotype.tif",
            dask=False,
        )
        np.testing.assert_array_equal(data.compute().values, data2.values)
    assert len(data.shape) == 5


@pytest.mark.io
def test_write_ome_zarr_image_dask(tmp_path):
    data = read_image(
        "scallops/tests/data/tif/10X_c0-DAPI-p65ab_A1_Tile-7.phenotype.tif", dask=True
    )
    data.attrs.clear()
    zarr_path = str(tmp_path / "test.zarr")
    _write_zarr_image("foo", open_ome_zarr(zarr_path), data)
    data2 = read_image(
        "scallops/tests/data/tif/10X_c0-DAPI-p65ab_A1_Tile-7.phenotype.tif", dask=False
    )
    np.testing.assert_array_equal(
        data2.values, data.values, err_msg="Dask and non dask images are not equal"
    )
    data3 = read_image(f"{zarr_path}/images/foo", dask=False)
    np.testing.assert_array_equal(
        data2.values, data3.values, err_msg="Saved image not equal"
    )


@pytest.mark.io
def test_write_non_ome_zarr_image(tmp_path, dask):
    image = read_image(
        "scallops/tests/data/tif/10X_c0-DAPI-p65ab_A1_Tile-7.phenotype.tif", dask=dask
    )
    image.attrs = {"test": "1"}
    image.attrs["physical_pixel_sizes"] = (1, 1, 1)
    image.attrs["physical_pixel_units"] = ("mm", "mm", "mm")
    zarr_path = str(tmp_path / "test.zarr")
    _write_zarr_image("img_zarr", open_ome_zarr(zarr_path), image, zarr_format="zarr")
    _write_zarr_image("img_ome_zarr", open_ome_zarr(zarr_path), image)

    data_zarr = read_image(f"{zarr_path}/images/img_zarr", dask=False)
    data_ome_zarr = read_image(f"{zarr_path}/images/img_ome_zarr", dask=False)

    xr.testing.assert_equal(data_zarr, data_ome_zarr)
    xr.testing.assert_equal(image, data_ome_zarr)


@pytest.mark.io
def test_experiment_file_list():
    image_paths = (
        "scallops/tests/data/experimentC/10X_c0-DAPI-p65ab/10X_c0-DAPI-p65ab_A1_Tile-102.phenotype.tif",
        "scallops/tests/data/experimentC/10X_c0-DAPI-p65ab/10X_c0-DAPI-p65ab_A1_Tile-",
    )

    gen = list(_set_up_experiment(image_paths, group_by=("*",)))
    assert len(gen) == 1
    x = gen[0]
    assert x[2]["id"] == "image", "Image id should be `image`"
    assert len(x[1]) == 2, "Should have 2 files"


@pytest.mark.io
def test_experiment_pattern_prefix(tmp_path):
    image_pattern = "c{t}/Well{well}_Point{point}_{tile}_Channel{channel}_Seq{seq}.tif"
    image_no_match = "xxx/Well{well}_Point{point}_{tile}_Channel{channel}_Seq{seq}.tif"
    src_path = "scallops/tests/data/experimentC/10X_c0-DAPI-p65ab/10X_c0-DAPI-p65ab_A1_Tile-102.phenotype.tif"
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
def test_experiment_separate_t_c(dask, tmp_path):
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
    image = read_experiment(tmp_path, pattern, group_by=("well",)).images["B06"]
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
                f"dask: {dask}, channel: {channel}, cycle: {cycle}, value: {value}",
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
def test_group_by_one_field(dask):
    exp = read_experiment(
        "scallops/tests/data/experimentC/input",
        "10X_c{t}-SBS-{t}/{mag}X_c{t}-{exp}-{t}_{well}_Tile-102.{datatype}.tif",
        group_by=("well",),
        dask=dask,
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
def test_read_write_labels(tmp_path, array_A1_102_nuclei):
    nuclei = array_A1_102_nuclei.squeeze().data

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
    )
    np.testing.assert_equal(
        exp.images["A1-102"].coords["t"].data, [1, 2, 3, 4, 5, 7, 8, 9, 10]
    )

    np.testing.assert_equal(
        exp.images["A1-102"].values, experiment_c.images["A1-102"].values
    )
    np.testing.assert_equal(
        exp.images["A1-103"].values, experiment_c.images["A1-103"].values
    )
    assert exp.images["A1-102"].attrs["common_src"] == "10X_c-SBS_*_A1_Tile-102.sbs"
    assert exp.images["A1-102"].attrs["group"] == dict(tile="102", well="A1")


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
            "scallops/tests/data/experimentC/input",
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
    exp = read_experiment(
        os.path.join("scallops", "tests", "data", "nd2"), file_pattern
    )
    path = os.path.join(tmp_path, "test.zarr")
    exp.save(path)
    exp = read_experiment(path)
    assert len(exp.images) == 1


@pytest.mark.io
def test_read_write_image_spacing_stack(tmp_path):
    image1 = read_image(
        "scallops/tests/data/experimentC/10X_c0-DAPI-p65ab/10X_c0-DAPI-p65ab_A1_Tile-102.phenotype.tif"
    ).squeeze()
    image2 = read_image(
        "scallops/tests/data/experimentC/10X_c0-DAPI-p65ab/10X_c0-DAPI-p65ab_A1_Tile-103.phenotype.tif"
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
        "scallops/tests/data/experimentC/10X_c0-DAPI-p65ab/10X_c0-DAPI-p65ab_A1_Tile-102.phenotype.tif"
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
            file_sort_order=["20231012_20x_6W_IF", "20231010_20x_6W_FISH"],
        )
    )[0]

    _, file_list, metadata = image_gen
    assert os.path.basename(file_list[0]) == "A1-20231012_20x_6W_IF.tiff"
    assert os.path.basename(file_list[1]) == "A1-20231010_20x_6W_FISH.tiff"


@pytest.mark.io
def test_read_experiment_csv(tmp_path):
    exp = read_experiment(
        "scallops/tests/data/experimentC/input",
        "10X_c{t}-SBS-{t}/{ignore}X_c{t}-{ignore}-{t}_{well}_Tile-{tile}.{ignore}.tif",
        group_by=("well", "tile"),
    )
    paths = glob.glob("scallops/tests/data/experimentC/input/**/*.tif")
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
