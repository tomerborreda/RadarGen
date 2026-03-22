"""Dataset adapters for various autonomous driving datasets."""

from radargen.datasets.registry import get_adapter, list_adapters, register_adapter

# Import adapter modules to trigger registration decorators
from radargen.datasets import truckscenes, nuscenes  # noqa: F401

__all__ = [
    "register_adapter",
    "get_adapter",
    "list_adapters",
]
