import os
import subprocess

import numpy as np
import pytest
import zarr

from scallops.io import read_image


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

    store = zarr.storage.ZipStore("scallops/tests/data/ops-illum-corr.zip", mode="r")
    root = zarr.open(store=store, mode="r")
    np.testing.assert_equal(
        root["data"][...],
        read_image(os.path.join(tmp_path, "images", "A1")).values.squeeze(),
    )
    # compare to known good result
