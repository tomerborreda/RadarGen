"""Core radar map creation pipeline using dataset adapters.

This module provides functions to create ground truth radar maps (PD, RCS, Doppler)
from radar point clouds for training the RadarGen diffusion model.
"""

import logging
import os
from typing import Any, Callable, List, Optional

import numpy as np
import torch

from radargen.core.protocols import DatasetAdapter
from radargen.radar_maps.config import RadarMapCreationConfig, RadarMapsCreationOutput
from radargen.radar_maps.map_creation import create_radar_pd_map, create_radar_rcs_map, create_radar_doppler_map

logger = logging.getLogger(__name__)


def filter_by_range(pcl: np.ndarray, point_limit: float) -> np.ndarray:
    """
    Filter point cloud by square range in the x-y plane.

    Keeps points where |x| < point_limit AND |y| < point_limit.

    Args:
        pcl: Point cloud array (N, 8+) with x,y in first 2 columns (meters)
        point_limit: Maximum absolute distance in meters

    Returns:
        Filtered point cloud (M, 8+) where M <= N
    """
    if pcl.shape[0] == 0:
        return pcl
    mask = (np.abs(pcl[:, 0]) < point_limit) & (np.abs(pcl[:, 1]) < point_limit)
    return pcl[mask].copy()


def normalize_to_image_coords(pcl: np.ndarray, point_limit: float, image_size: int) -> np.ndarray:
    """
    Convert x,y from meters to pixel coordinates.

    Maps [-point_limit, point_limit] meters to [0, image_size] pixels.
    Other columns are unchanged.

    Args:
        pcl: Point cloud array (N, 8+) with x,y in meters
        point_limit: Spatial range in meters
        image_size: Target image size in pixels

    Returns:
        Point cloud with x,y converted to pixel coordinates
    """
    if pcl.shape[0] == 0:
        return pcl
    result = pcl.copy()
    result[:, 0] = (result[:, 0] + point_limit) / (2 * point_limit) * image_size
    result[:, 1] = (result[:, 1] + point_limit) / (2 * point_limit) * image_size
    return result


def deduplicate_by_pixel(pcl: np.ndarray) -> np.ndarray:
    """
    Keep only the highest-RCS point per pixel location.

    Rounds x,y to integer pixel coordinates and keeps the point with
    the highest RCS (column 6) at each unique pixel.

    Args:
        pcl: Point cloud array (N, 8+) with x,y in pixel coordinates

    Returns:
        Deduplicated point cloud (M, 8+) where M <= N
    """
    if pcl.shape[0] == 0:
        return pcl

    pixel_coords = np.round(pcl[:, :2]).astype(np.int32)

    unique_coords, inverse_indices = np.unique(
        pixel_coords, axis=0, return_inverse=True
    )

    unique_pc_list = []
    for i in range(len(unique_coords)):
        point_indices = np.where(inverse_indices == i)[0]
        best_idx = point_indices[np.argmax(pcl[point_indices, 6])]
        unique_pc_list.append(pcl[best_idx])

    if len(unique_pc_list) == 0:
        return np.zeros((0, pcl.shape[1]), dtype=pcl.dtype)

    return np.vstack(unique_pc_list)


def create_radar_maps_for_frame(
        adapter: DatasetAdapter,
        sample_data: dict,
        device: torch.device,
        image_size: int = 512,
        point_limit: float = 50.0,
        nsweeps: int = 1,
        sigma_range: tuple = (2.0, 2.0),
        filter_camera_views: Optional[List[str]] = None,
) -> RadarMapsCreationOutput:
    """
    Create radar maps (PD, RCS, Doppler) for a single frame.

    Pipeline:
    1. Load radar point cloud from adapter
    2. Filter by camera frustum
    3. Filter by square range
    4. Normalize to image coordinates
    5. Deduplicate by pixel
    6. Create PD, RCS, Doppler maps

    Args:
        adapter: Dataset adapter instance
        sample_data: Dictionary containing sample data tokens
        device: torch device for computation
        image_size: Size of the output maps
        point_limit: Maximum distance from ego vehicle (meters)
        nsweeps: Number of radar sweeps to aggregate
        sigma_range: Sigma values for Gaussian splatting
        filter_camera_views: Optional list of camera view names to filter by frustum

    Returns:
        RadarMapsGenerationOutput containing PD, RCS, and Doppler maps
    """
    # 1. Load the fused radar pointcloud (in flat-up frame)
    pcl = adapter.load_radar_pointcloud(sample_data, nsweeps=nsweeps)
    logger.debug(f"Loaded {pcl.shape[0]} radar points")

    # 2. Get ref→flat-up transform for camera frustum filtering
    # Points from load_radar_pointcloud are in flat-up frame, so we need to
    # transform cameras to flat-up for correct frustum filtering
    ref_sensor_token = sample_data.get(adapter.reference_sensor)
    if ref_sensor_token is None:
        raise ValueError(f"{adapter.reference_sensor} not found in sample_data")

    ref_to_flat_up = adapter.get_ref_to_vehicle_flat_up_transform(ref_sensor_token)

    # 3. Filter by camera frustum (points are in flat-up, so transform cameras to flat-up)
    pcl = adapter.filter_by_camera_frustum(
        pcl,
        sample_data,
        camera_views=filter_camera_views,
        ref_to_points_transform=ref_to_flat_up
    )
    logger.debug(f"After frustum filter: {pcl.shape[0]} points")

    # 3. Filter by square range
    pcl = filter_by_range(pcl, point_limit)
    logger.debug(f"After range filter: {pcl.shape[0]} points (limit={point_limit}m)")

    # 4. Normalize to image coordinates
    pcl = normalize_to_image_coords(pcl, point_limit, image_size)

    # 5. Deduplicate by pixel
    pcl = deduplicate_by_pixel(pcl)
    logger.debug(f"After dedup: {pcl.shape[0]} points")

    # Validate that we have points after filtering
    if pcl.shape[0] == 0:
        raise RuntimeError(
            f"Empty point cloud after filtering for sample {sample_data}. "
            f"This indicates an upstream issue:\n"
            f"  - All points filtered out (check point_limit={point_limit})\n"
            f"  - All points outside camera frustums\n"
            f"  - No radar data in scene\n"
            f"Please investigate the filtering parameters or scene data."
        )

    # Extract data for map creation
    points_transposed = pcl.T
    locations_2d = points_transposed[:2, :]  # (2, N) - xy coordinates
    rcs_values = points_transposed[6, :]  # (N,) - RCS values
    radial_velocity = points_transposed[7, :]  # (N,) - radial velocity

    # Create maps
    norm = adapter.normalization
    pd_map = create_radar_pd_map(locations_2d, image_size, device=device, sigma_range=sigma_range)
    rcs_map = create_radar_rcs_map(locations_2d, rcs_values, image_size, rcs_min=norm.rcs_min, rcs_max=norm.rcs_max)
    doppler_map = create_radar_doppler_map(locations_2d, radial_velocity, image_size)

    return RadarMapsCreationOutput(pd_map, rcs_map, doppler_map)


def create_and_save_radar_maps_for_frame(
        adapter: DatasetAdapter,
        scene_token: str,
        frame_idx: int,
        sample_data: dict,
        save_dir_path: str,
        device: torch.device,
        image_size: int = 512,
        point_limit: float = 50.0,
        nsweeps: int = 1,
        sigma_range: tuple = (2.0, 2.0),
        filter_camera_views: Optional[List[str]] = None,
):
    """
    Create and save radar maps for a single frame to disk.

    Args:
        adapter: Dataset adapter instance
        scene_token: Scene token to process
        frame_idx: Frame index to process
        sample_data: Dictionary containing sample data tokens
        save_dir_path: Directory to save the maps
        device: torch device for computation
        image_size: Size of the output maps
        point_limit: Maximum distance from ego vehicle (meters)
        nsweeps: Number of radar sweeps to aggregate
        sigma_range: Sigma values for Gaussian splatting
        filter_camera_views: Optional list of camera view names to filter
    """
    map_creation_output = create_radar_maps_for_frame(
        adapter=adapter,
        sample_data=sample_data,
        device=device,
        image_size=image_size,
        point_limit=point_limit,
        nsweeps=nsweeps,
        sigma_range=sigma_range,
        filter_camera_views=filter_camera_views,
    )

    # Save radar point density map
    pd_map_path = os.path.join(save_dir_path, f'radar_pd_map_{scene_token}_{frame_idx}')
    np.save(pd_map_path, map_creation_output.pd_map)

    # Save radar RCS map
    rcs_map_path = os.path.join(save_dir_path, f'radar_rcs_map_{scene_token}_{frame_idx}')
    np.save(rcs_map_path, map_creation_output.rcs_map)

    # Save radar Doppler map
    doppler_map_path = os.path.join(save_dir_path, f'radar_doppler_map_{scene_token}_{frame_idx}')
    np.save(doppler_map_path, map_creation_output.doppler_map)


def create_radar_maps_for_scene(
        adapter: DatasetAdapter,
        scene_token: str,
        samples: List[Any],
        save_dir_path: str,
        device: torch.device,
        image_size: int = 512,
        point_limit: float = 50.0,
        nsweeps: int = 1,
        sigma_range: tuple = (2.0, 2.0),
        filter_camera_views: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
):
    """
    Create radar maps for all frames in a scene.

    Args:
        adapter: Dataset adapter instance
        scene_token: Scene token to process
        samples: List of sample data for the scene (from iter_scenes)
        save_dir_path: Directory to save the maps
        device: torch device for computation
        image_size: Size of the output maps
        point_limit: Maximum distance from ego vehicle (meters)
        nsweeps: Number of radar sweeps to aggregate
        sigma_range: Sigma values for Gaussian splatting
        filter_camera_views: Optional list of camera view names to filter
        progress_callback: Optional callback(sample_idx, total_samples) for progress updates
    """
    # Process samples and update progress via callback
    total_samples = len(samples)
    for frame_idx, sample_data in enumerate(samples):
        # Update progress before processing
        if progress_callback:
            progress_callback(frame_idx, total_samples)

        # sample_data is a dict of sensor tokens (e.g., {"CAMERA_LEFT_FRONT": "token1", ...})
        create_and_save_radar_maps_for_frame(
            adapter=adapter,
            scene_token=scene_token,
            frame_idx=frame_idx,
            sample_data=sample_data,
            save_dir_path=save_dir_path,
            device=device,
            image_size=image_size,
            point_limit=point_limit,
            nsweeps=nsweeps,
            sigma_range=sigma_range,
            filter_camera_views=filter_camera_views,
        )

    # Final update when scene completes
    if progress_callback:
        progress_callback(total_samples - 1, total_samples)
