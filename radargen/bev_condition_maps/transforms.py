"""Coordinate transformation utilities for BEV map generation."""

from typing import Dict, Any, List
import numpy as np

from radargen.core.protocols import DatasetAdapter


def transform_depth_points_to_bev(
    adapter: DatasetAdapter,
    depth_outputs: List[dict],
    sample_data: Dict[str, Any],
    camera_views: List[str],
    ref_sensor_token: str,
) -> np.ndarray:
    """Transform depth points from all cameras to ego BEV frame.

    Args:
        adapter: Dataset adapter providing coordinate transformations
        depth_outputs: List of depth output dicts (one per camera) with 'points' key
        sample_data: Sample data dict for the current frame
        camera_views: List of camera view names
        ref_sensor_token: Reference sensor token

    Returns:
        (N, 3) array of points in vehicle flat-up BEV coordinates (x, y, z)
    """
    # Get transformations from adapter
    camera_extrinsics = adapter.get_camera_extrinsics(sample_data, ref_sensor_token)
    ref_to_flat_up = adapter.get_ref_to_vehicle_flat_up_transform(ref_sensor_token)

    # Transform each camera's points
    all_points = []
    for depth_output, extrinsic in zip(depth_outputs, camera_extrinsics):
        # Extract points from depth output (shape: [H, W, 3])
        points = depth_output['points'].reshape(-1, 3).cpu().numpy()

        # Transform: Camera → Reference
        points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
        points_ref = extrinsic.dot(points_homogeneous.T).T[:, :3]

        # Transform: Reference → Vehicle Flat-Up BEV
        points_ref_homogeneous = np.hstack([points_ref, np.ones((points_ref.shape[0], 1))])
        points_flat_up = ref_to_flat_up.dot(points_ref_homogeneous.T).T[:, :3]

        all_points.append(points_flat_up)

    return np.concatenate(all_points, axis=0)
