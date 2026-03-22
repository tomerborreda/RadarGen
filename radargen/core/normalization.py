"""Normalization configuration for radar data across different datasets."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class NormalizationConfig:
    """
    Configuration for normalizing radar data values.

    Different datasets may have different value ranges for RCS, Doppler, and
    coordinate ranges. This class provides a single place to configure these
    values per dataset.

    Attributes:
        coordinate_range: Spatial range in meters (e.g., 50.0 means [-50, 50] in x and y)
        rcs_min: Minimum RCS value in dBm² (default: -20.0)
        rcs_max: Maximum RCS value in dBm² (default: 66.0)
        doppler_min: Minimum Doppler velocity in m/s (default: -120.0)
        doppler_max: Maximum Doppler velocity in m/s (default: 120.0)
        camera_freq: Camera capture frequency in Hz (default: 10.0)
    """
    coordinate_range: float = 50.0
    rcs_min: float = -20.0
    rcs_max: float = 66.0
    doppler_min: float = -120.0
    doppler_max: float = 120.0
    camera_freq: float = 10.0

    @property
    def xy_range(self) -> Tuple[float, float]:
        """Return the (min, max) coordinate range."""
        return (-self.coordinate_range, self.coordinate_range)

    @property
    def rcs_range(self) -> Tuple[float, float]:
        """Return the (min, max) RCS range."""
        return (self.rcs_min, self.rcs_max)

    @property
    def doppler_range(self) -> Tuple[float, float]:
        """Return the (min, max) Doppler range."""
        return (self.doppler_min, self.doppler_max)

    def normalize_rcs(self, rcs: float) -> float:
        """Normalize RCS value to [0, 1] range."""
        return (rcs - self.rcs_min) / (self.rcs_max - self.rcs_min)

    def denormalize_rcs(self, normalized: float) -> float:
        """Convert normalized [0, 1] RCS back to original scale."""
        return normalized * (self.rcs_max - self.rcs_min) + self.rcs_min

    def normalize_doppler(self, doppler: float) -> float:
        """Normalize Doppler value to [0, 1] range."""
        return (doppler - self.doppler_min) / (self.doppler_max - self.doppler_min)

    def denormalize_doppler(self, normalized: float) -> float:
        """Convert normalized [0, 1] Doppler back to original scale."""
        return normalized * (self.doppler_max - self.doppler_min) + self.doppler_min

    def normalize_coordinates(self, x: float, y: float) -> Tuple[float, float]:
        """Normalize coordinates to [0, 1] range based on coordinate_range."""
        x_norm = (x + self.coordinate_range) / (2 * self.coordinate_range)
        y_norm = (y + self.coordinate_range) / (2 * self.coordinate_range)
        return x_norm, y_norm

    def denormalize_coordinates(self, x_norm: float, y_norm: float) -> Tuple[float, float]:
        """Convert normalized [0, 1] coordinates back to original scale."""
        x = x_norm * (2 * self.coordinate_range) - self.coordinate_range
        y = y_norm * (2 * self.coordinate_range) - self.coordinate_range
        return x, y


# Default normalization matching current TruckScenes configuration
default_normalization = NormalizationConfig(
    coordinate_range=50.0,
    rcs_min=-20.0,
    rcs_max=66.0,
    doppler_min=-120.0,
    doppler_max=120.0,
    camera_freq=10.0,
)


# Pre-defined configurations for known datasets
TRUCKSCENES_NORMALIZATION = NormalizationConfig(
    coordinate_range=50.0,
    rcs_min=-20.0,
    rcs_max=66.0,
    doppler_min=-120.0,
    doppler_max=120.0,
    camera_freq=10.0,
)

NUSCENES_NORMALIZATION = NormalizationConfig(
    coordinate_range=50.0,
    rcs_min=-20.0,
    rcs_max=66.0,
    doppler_min=-120.0,
    doppler_max=120.0,
    camera_freq=12.0,  # nuScenes has 12Hz cameras
)
