"""TruckScenes-specific configuration constants."""

from radargen.core.normalization import NormalizationConfig
from radargen.core.protocols import DatasetConfig

# TruckScenes camera view names
TRUCKSCENES_CAMERA_VIEWS = [
    "CAMERA_RIGHT_FRONT",
    "CAMERA_LEFT_FRONT",
    "CAMERA_RIGHT_BACK",
    "CAMERA_LEFT_BACK",
]

# Reference sensor used for coordinate transforms
TRUCKSCENES_REFERENCE_SENSOR = "LIDAR_LEFT"

# Radar sensors in TruckScenes
TRUCKSCENES_RADAR_SENSORS = [
    "RADAR_LEFT_FRONT",
    "RADAR_RIGHT_FRONT",
    "RADAR_LEFT_SIDE",
    "RADAR_RIGHT_SIDE",
    "RADAR_LEFT_BACK",
    "RADAR_RIGHT_BACK",
]

# TruckScenes normalization configuration
TRUCKSCENES_NORMALIZATION = NormalizationConfig(
    coordinate_range=50.0,
    rcs_min=-20.0,
    rcs_max=66.0,
    doppler_min=-120.0,
    doppler_max=120.0,
    camera_freq=10.0,
)

# Default TruckScenes configuration
TRUCKSCENES_CONFIG = DatasetConfig(
    name="truckscenes",
    camera_views=TRUCKSCENES_CAMERA_VIEWS,
    reference_sensor=TRUCKSCENES_REFERENCE_SENSOR,
    normalization=TRUCKSCENES_NORMALIZATION,
    version="v1.0-trainval",
)

# Scene token that should be excluded (missing camera token)
EXCLUDED_SCENE_INDEX = 345
