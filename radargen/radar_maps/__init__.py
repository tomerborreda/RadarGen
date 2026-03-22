"""Radar map creation package for RadarGen.

This package provides functionality for creating ground truth radar maps (Point Density,
RCS, and Doppler) from radar point clouds for training the RadarGen diffusion model.

Main modules:
- config: Configuration dataclasses and constants
- creator: Core creation pipeline
- map_creation: PD, RCS, and Doppler map creation functions
- transforms: Coordinate transformation utilities

"""

# Configuration and data structures
from .config import (
    RadarMapCreationConfig,
    RadarMapsCreationOutput,
)

# Core creation functions
from .creator import (
    filter_by_range,
    normalize_to_image_coords,
    deduplicate_by_pixel,
    create_radar_maps_for_frame,
    create_and_save_radar_maps_for_frame,
    create_radar_maps_for_scene,
)

# Map creation functions
from .map_creation import (
    # Point density
    create_radar_pd_map,
    decoded_to_pd_map,
    pd_map_to_image,
    point_cloud_to_point_density_map,
    # RCS
    create_interpolated_rcs_map,
    create_radar_rcs_map,
    create_raw_rcs_map,
    decoded_to_rcs_map,
    rcs_map_to_image,
    # Doppler
    create_interpolated_doppler_map,
    create_radar_doppler_map,
    create_raw_doppler_map,
    decoded_to_doppler_map,
    doppler_map_to_image,
)

__all__ = [
    # Config
    "RadarMapCreationConfig",
    "RadarMapsCreationOutput",
    # Creation utilities
    "filter_by_range",
    "normalize_to_image_coords",
    "deduplicate_by_pixel",
    # Creation functions
    "create_radar_maps_for_frame",
    "create_and_save_radar_maps_for_frame",
    "create_radar_maps_for_scene",
    # Point density
    "create_radar_pd_map",
    "decoded_to_pd_map",
    "pd_map_to_image",
    "point_cloud_to_point_density_map",
    # RCS
    "create_interpolated_rcs_map",
    "create_radar_rcs_map",
    "create_raw_rcs_map",
    "decoded_to_rcs_map",
    "rcs_map_to_image",
    # Doppler
    "create_interpolated_doppler_map",
    "create_radar_doppler_map",
    "create_raw_doppler_map",
    "decoded_to_doppler_map",
    "doppler_map_to_image",
]
