"""nuScenes-specific configuration constants."""

from radargen.core.normalization import NormalizationConfig
from radargen.core.protocols import DatasetConfig

# nuScenes camera view names (6 surround-view cameras)
NUSCENES_CAMERA_VIEWS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

# Reference sensor used for coordinate transforms
NUSCENES_REFERENCE_SENSOR = "LIDAR_TOP"

# Radar sensors in nuScenes (5 sensors)
NUSCENES_RADAR_SENSORS = [
    "RADAR_FRONT",
    "RADAR_FRONT_LEFT",
    "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT",
    "RADAR_BACK_RIGHT",
]

NUSCENES_NORMALIZATION = NormalizationConfig(
    coordinate_range=50.0,
    rcs_min=-20.0,
    rcs_max=66.0,
    doppler_min=-120.0,
    doppler_max=120.0,
    camera_freq=12.0,  # nuScenes has 12Hz cameras
)

# Default nuScenes configuration
NUSCENES_CONFIG = DatasetConfig(
    name="nuscenes",
    camera_views=NUSCENES_CAMERA_VIEWS,
    reference_sensor=NUSCENES_REFERENCE_SENSOR,
    normalization=NUSCENES_NORMALIZATION,
    version="v1.0-trainval",
)
