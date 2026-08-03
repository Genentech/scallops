import warnings

from .experiment.elements import Experiment  # noqa: F401

warnings.filterwarnings(
    "ignore", message="unclosed.*", category=ResourceWarning, module="aiohttp"
)
