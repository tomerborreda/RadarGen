"""Configuration and data structures for radar map creation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class RadarMapCreationConfig:
    """Configuration for creating the ground truth radar maps used for training."""

    # Dataset name (e.g., 'truckscenes', 'nuscenes')
    dataset_name: str
    # Directory containing the dataset
    dataset_dir: Path
    # Directory to save the radar maps to
    save_dir: Path
    # Image size for the radar maps
    image_size: int = field(default=512)
    # Point distance threshold for the radar maps (meters)
    point_limit: float = field(default=50.0)
    # Number of sweeps for the radar maps
    nsweeps: int = field(default=1)
    # Sigma for the Gaussian kernel of the radar maps
    sigma: float = field(default=2.0)
    # Version of the dataset (e.g., 'v1.0-trainval', 'v1.0-mini')
    dataset_version: str = field(default='v1.0-trainval')
    # Filter for the camera views. None for all views.
    filter_camera_views: Optional[List[str]] = None
    # Filter for the split. None for all splits. Choose between 'train' and 'val'.
    filter_for_split: Optional[str] = field(default=None)


@dataclass
class RadarMapsCreationOutput:
    """Output structure for radar map creation."""

    pd_map: np.ndarray
    rcs_map: np.ndarray
    doppler_map: np.ndarray
