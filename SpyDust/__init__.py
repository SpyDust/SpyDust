"""Spinning-dust emission and grain dynamics."""

from importlib import import_module
import os

__version__ = "1.4.0"
package_version = __version__
SpDust_data_dir = os.path.join(os.path.dirname(__file__), "Data_Files", "")

_submodules = (
    "AngMomDist", "charge_dist", "collisions", "free_free", "Grain",
    "H2_photoemission", "infrared", "mpiutil", "plasmadrag", "SED",
    "SpyDust", "util", "mom_exp", "SED_fit", "full_Stokes",
)
__all__ = ["SpDust_data_dir", "package_version", *_submodules]


def __getattr__(name):
    if name in _submodules:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_submodules))
