import os
import subprocess

import numpy as np
import pytest
import zarr
from zarr.storage import ZipStore

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

    with ZipStore("scallops/tests/data/ops-illum-corr.zip", read_only=True) as store:
        root = zarr.open(store=store)
        # compare to known good result
        np.testing.assert_equal(
            root["data"][...],
            read_image(os.path.join(tmp_path, "images", "A1")).values.squeeze(),
        )
