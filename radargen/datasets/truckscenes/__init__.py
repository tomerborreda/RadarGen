"""TruckScenes dataset adapter for RadarGen."""

from radargen.datasets.truckscenes.adapter import TruckScenesAdapter
from radargen.datasets.truckscenes.config import (
    TRUCKSCENES_CAMERA_VIEWS,
    TRUCKSCENES_CONFIG,
    TRUCKSCENES_REFERENCE_SENSOR,
)

__all__ = [
    "TruckScenesAdapter",
    "TRUCKSCENES_CONFIG",
    "TRUCKSCENES_CAMERA_VIEWS",
    "TRUCKSCENES_REFERENCE_SENSOR",
]
