"""Converter from MTG/FCI contrail labels and images to the DeepContrail dataset layout."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("deepcontrail-mtg")
except PackageNotFoundError:  # used from a source checkout without installing
    __version__ = "0+unknown"

__all__ = ["__version__"]
