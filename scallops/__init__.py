import warnings

import anndata
import zarr

from scallops.zarr_io import write_basic_dask_dask_dense  # noqa: F401

from .experiment.elements import Experiment  # noqa: F401

warnings.filterwarnings(
    "ignore",
    # the message is matched with re.match, so each alternative has to match from the
    # start and must not have stray whitespace around the "|"
    message="Unclosed client.*|.*client_session.*",
    category=ResourceWarning,
)
anndata.settings.auto_shard_zarr_v3 = False
zarr.config.set({"array.rectilinear_chunks": True})
