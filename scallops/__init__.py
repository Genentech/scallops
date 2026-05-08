import logging
import warnings

from .experiment.elements import Experiment  # noqa: F401

logging.getLogger("asyncio").setLevel(logging.CRITICAL)
warnings.filterwarnings(
    "ignore",
    message="Unclosed client session",
    category=ResourceWarning,
    module="aiohttp",
)
warnings.filterwarnings(
    "ignore",
    message="Writing zarr v2 data will no longer be the default",
    category=UserWarning,
    module="anndata",
)
