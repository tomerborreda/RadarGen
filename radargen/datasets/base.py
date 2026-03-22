"""Base utilities for dataset adapters."""

from typing import Any, Dict, List, Optional, Tuple, TypeVar

import numpy as np

from radargen.core.data_types import RadarGenInferenceInput
from radargen.core.protocols import DatasetAdapter

T = TypeVar('T')


def build_inference_input(
    adapter: DatasetAdapter,
    sample_data: Dict[str, Any],
    ref_sensor_token: Optional[str] = None,
    images: Optional[List[np.ndarray]] = None,
) -> RadarGenInferenceInput:
    """
    Build RadarGenInferenceInput from a dataset adapter and sample data.

    This is a convenience function that extracts all necessary information
    using the adapter interface.

    Args:
        adapter: Dataset adapter instance
        sample_data: Sample data dictionary from the adapter
        ref_sensor_token: Optional reference sensor token. If None, uses default.
        images: Optional pre-loaded images. If None, images are loaded.

    Returns:
        RadarGenInferenceInput ready for inference
    """
    if ref_sensor_token is None:
        ref_sensor_token = adapter.get_reference_sensor_token(sample_data)

    if images is None:
        images = adapter.load_camera_images(sample_data)

    intrinsics = adapter.get_camera_intrinsics(sample_data)
    extrinsics = adapter.get_camera_extrinsics(sample_data, ref_sensor_token)
    ref_to_flat_up = adapter.get_ref_to_vehicle_flat_up_transform(ref_sensor_token)

    return RadarGenInferenceInput(
        input_images=images,
        camera_intrinsics=intrinsics,
        camera_extrinsics=extrinsics,
        ref_to_ego_flat_up=ref_to_flat_up,
    )


def validate_sample_data(
    adapter: DatasetAdapter,
    sample_data: Dict[str, Any],
) -> bool:
    """
    Validate that sample data contains all required sensor tokens.

    Args:
        adapter: Dataset adapter instance
        sample_data: Sample data dictionary to validate

    Returns:
        True if valid, raises ValueError otherwise
    """
    missing = []

    # Check camera views
    for camera in adapter.camera_views:
        if camera not in sample_data:
            missing.append(camera)

    # Check reference sensor
    if adapter.reference_sensor not in sample_data:
        missing.append(adapter.reference_sensor)

    if missing:
        raise ValueError(
            f"Sample data missing required sensors: {missing}. "
            f"Available: {list(sample_data.keys())}"
        )

    return True


def get_sample_token_from_data(
    sample_data: Dict[str, Any],
    sensor_key: str,
) -> str:
    """
    Extract a sensor token from sample data.

    Args:
        sample_data: Sample data dictionary
        sensor_key: Key for the sensor token

    Returns:
        Sensor token string

    Raises:
        KeyError: If sensor_key not in sample_data
    """
    if sensor_key not in sample_data:
        raise KeyError(
            f"Sensor '{sensor_key}' not found in sample data. "
            f"Available: {list(sample_data.keys())}"
        )
    return sample_data[sensor_key]


def distribute_across_ranks(
    items: List[T],
    rank: int,
    world_size: int,
) -> List[T]:
    """
    Distribute items across ranks using stride pattern.

    This is useful for multi-GPU distributed processing where each GPU
    should process a subset of the data. The stride pattern ensures
    that items are distributed evenly across ranks.

    Args:
        items: List of items to distribute
        rank: Current rank (0-indexed)
        world_size: Total number of ranks

    Returns:
        Subset of items for this rank (every Nth item starting at rank)

    Example:
        >>> # Distribute scenes across 8 GPUs
        >>> scenes = list(adapter.iter_scenes('train'))
        >>> local_scenes = distribute_across_ranks(scenes, rank=0, world_size=8)
        >>> # rank 0 gets scenes 0, 8, 16, ...
        >>> # rank 1 gets scenes 1, 9, 17, ...
        >>> for scene_token, samples in local_scenes:
        ...     process_scene(scene_token, samples)

        >>> # Distribute consecutive pairs across 4 GPUs
        >>> all_pairs = list(adapter.iter_consecutive_pairs('train'))
        >>> local_pairs = distribute_across_ranks(all_pairs, rank=2, world_size=4)
        >>> # rank 2 gets pairs 2, 6, 10, 14, ...
    """
    return items[rank::world_size]
