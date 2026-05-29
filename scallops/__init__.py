import warnings

from .experiment.elements import Experiment  # noqa: F401

warnings.filterwarnings(
    "ignore",
    message="Writing zarr v2 data will no longer be the default",
    category=UserWarning,
    module="anndata",
)

warnings.filterwarnings(
    "ignore",
    message="anndata will no longer support zarr v2 in the near future",
    category=DeprecationWarning,
    module="anndata",
)
