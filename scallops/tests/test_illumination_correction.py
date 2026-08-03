import os
import subprocess

import numpy as np
import pytest
import tifffile
import zarr
from zarr.storage import ZipStore

from scallops.cli.illumination_correction import single_agg_illumination_correction
from scallops.io import read_image
from scallops.zarr_io import open_ome_zarr


@pytest.mark.io
def test_illumination_correction_cli(tmp_path):
    tmp_path = str(tmp_path / "test.zarr")
    args = [
        "scallops",
        "illum-corr",
        "agg",
        "--images",
        "scallops/tests/data/experimentC/input/10X_c1-SBS-1",
        "--image-pattern",
        "{mag}X_c{t}-{exp}-{t}_{well}_Tile-{tile}.{datatype}.tif",
        "--output-image-format",
        "zarr",
        "--groupby",
        "well",
        "-o",
        tmp_path,
    ]
    subprocess.check_call(args)

    with ZipStore("scallops/tests/data/ops-illum-corr.zip", mode="r") as store:
        root = zarr.open(store=store)
        np.testing.assert_equal(
            root["data"][...],
            read_image(os.path.join(tmp_path, "images", "A1")).values.squeeze(),
        )
    # compare to known good result


@pytest.mark.io
def test_illumination_correction_t_index(tmp_path):
    # synthetic multi-timepoint tiles: dims (t=3, c=2, y=32, x=32), each t offset by
    # a known amount so the selected timepoint is verifiable from the output.
    rng = np.random.default_rng(0)
    files = []
    for tile in range(2):
        arr = (rng.random((3, 2, 32, 32)) * 1000).astype(np.uint16)
        for t in range(3):
            arr[t] += t * 100
        path = tmp_path / f"A1_Tile-{tile}.tif"
        tifffile.imwrite(path, arr, metadata={"axes": "TCYX"})
        files.append(str(path))
    image_tuple = (("A1",), files, {"id": "A1"})

    def run(t_index):
        out_path = tmp_path / f"out_{t_index}.zarr"
        root = open_ome_zarr(str(out_path), mode="a")
        single_agg_illumination_correction(
            image_tuple=image_tuple,
            root=root,
            output_image_format="zarr",
            rescale=False,
            smooth=0,
            force=True,
            t_index=t_index,
        )
        return read_image(os.path.join(str(out_path), "images", "A1")).values

    with pytest.raises(ValueError, match="timepoints"):
        run(None)

    image_t0 = run(0)
    image_t2 = run(2)
    assert image_t0.shape == (2, 32, 32)
    assert image_t2.shape == (2, 32, 32)
    # t=2 has a larger additive offset than t=0, so its mean should be higher.
    assert image_t0.mean() < image_t2.mean()
