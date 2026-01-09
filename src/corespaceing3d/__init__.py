from .utils import main as main
from .utils import projection_correction as projection_correction
from .utils import run_main_mc as run_main_mc

__all__ = ["core", "tasks", "plotting", "utils", "main", "projection_correction", "run_main_mc"]

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version(__package__ or __name__)
except PackageNotFoundError:
    # Fallback
    try:
        from ._version import version as __version__
    except ImportError:
        __version__ = "0.0.0"
