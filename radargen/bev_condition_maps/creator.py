"""BEV map creation pipeline with multi-dataset support."""

import concurrent.futures
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from radargen.core.protocols import DatasetAdapter
from .core import create_bev_maps_from_camera_data
from .transforms import transform_depth_points_to_bev

# Thread pool for async I/O
_io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


def create_bev_maps_for_frame(
    adapter: DatasetAdapter,
    sample_data_t0: Dict[str, Any],
    sample_data_t1: Dict[str, Any],
    models: dict,
    device: torch.device,
    resolution: int = 512,
    use_batched_inference: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Create BEV color, segmentation, and velocity maps for a frame.

    Args:
        adapter: Dataset adapter (TruckScenesAdapter or NuScenesAdapter)
        sample_data_t0: Sample data at time t0
        sample_data_t1: Sample data at time t1
        models: Dict with foundation models from load_models()
        device: PyTorch device
        resolution: Output resolution (coordinate_range comes from adapter)
        use_batched_inference: If True, use batched model inference for speedup

    Returns:
        Tuple of (bev_color_map, bev_seg_map, bev_velocity_map)
        - bev_color_map: (resolution, resolution, 3) RGB color map
        - bev_seg_map: (resolution, resolution, 3) Segmentation map
        - bev_velocity_map: (resolution, resolution, 1) Velocity map
    """
    # Get coordinate range from adapter normalization config
    coordinate_range = adapter.normalization.coordinate_range

    # Compute per-camera time deltas from hardware timestamps
    timestamps_t0 = adapter.get_camera_timestamps(sample_data_t0)
    timestamps_t1 = adapter.get_camera_timestamps(sample_data_t1)
    dts = [abs(ts1 - ts0) / 1e6 for ts0, ts1 in zip(timestamps_t0, timestamps_t1)]

    # Load camera images using adapter
    camera_images_t0 = adapter.load_camera_images(sample_data_t0)
    camera_images_t1 = adapter.load_camera_images(sample_data_t1)
    camera_intrinsics = adapter.get_camera_intrinsics(sample_data_t0)

    # Create transformation function for this adapter
    ref_sensor_token = sample_data_t0[adapter.reference_sensor]

    def transform_fn(depth_outputs: List[dict]) -> np.ndarray:
        return transform_depth_points_to_bev(
            adapter, depth_outputs, sample_data_t0,
            adapter.camera_views, ref_sensor_token
        )

    # Generate BEV maps using shared logic
    norm = adapter.normalization
    bev_color_map, bev_seg_map, bev_velocity_map = create_bev_maps_from_camera_data(
        camera_images_t0=camera_images_t0,
        camera_images_t1=camera_images_t1,
        camera_intrinsics=camera_intrinsics,
        transform_fn=transform_fn,
        models=models,
        device=device,
        dts=dts,
        resolution=resolution,
        coordinate_range=coordinate_range,
        use_batched_inference=use_batched_inference,
        doppler_min=norm.doppler_min,
        doppler_max=norm.doppler_max,
    )

    return bev_color_map, bev_seg_map, bev_velocity_map


def create_bev_maps_for_scene(
    adapter: DatasetAdapter,
    scene_token: str,
    samples: List[Any],
    models: dict,
    save_dir: Path,
    device: torch.device,
    resolution: int = 512,
    use_batched_inference: bool = True,
    use_async_io: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
):
    """Create and save BEV maps for all frames in a scene.

    Args:
        adapter: Dataset adapter
        scene_token: Scene identifier token
        samples: List of sample data for the scene (from iter_scenes)
        models: Dict with foundation models
        save_dir: Directory to save generated maps
        device: PyTorch device
        resolution: Output resolution
        use_batched_inference: If True, use batched model inference for speedup
        use_async_io: If True, use async disk writes for improved I/O performance
        progress_callback: Optional callback(sample_idx, total_samples) for progress updates
    """
    # Helper function to save maps in background thread
    def _save_maps_async(save_dir: Path, scene_token: str, frame_idx: int, bev_color, bev_seg, bev_velocity):
        """Save BEV maps to disk."""
        np.save(save_dir / f'bev_color_map_{scene_token}_{frame_idx}.npy', bev_color)
        np.save(save_dir / f'bev_seg_map_{scene_token}_{frame_idx}.npy', bev_seg)
        np.save(save_dir / f'bev_velocity_map_{scene_token}_{frame_idx}.npy', bev_velocity)

    # Process samples and update progress via callback
    total_samples = len(samples) - 1  # We use pairs: t0, t1
    pending_writes = []

    for i in range(total_samples):
        # Update progress before processing
        if progress_callback:
            progress_callback(i, total_samples)

        sample_t0 = samples[i]
        sample_t1 = samples[i + 1]

        bev_color, bev_seg, bev_velocity = create_bev_maps_for_frame(
            adapter, sample_t0, sample_t1, models, device, resolution,
            use_batched_inference
        )

        if use_async_io:
            # Submit to thread pool (non-blocking)
            future = _io_executor.submit(
                _save_maps_async, save_dir, scene_token, i, bev_color, bev_seg, bev_velocity
            )
            pending_writes.append(future)
        else:
            # Original synchronous writes
            np.save(save_dir / f'bev_color_map_{scene_token}_{i}.npy', bev_color)
            np.save(save_dir / f'bev_seg_map_{scene_token}_{i}.npy', bev_seg)
            np.save(save_dir / f'bev_velocity_map_{scene_token}_{i}.npy', bev_velocity)

    # Wait for all writes to complete before returning
    if use_async_io:
        for future in pending_writes:
            future.result()  # Block until write completes

    # Final update when scene completes
    if progress_callback:
        progress_callback(total_samples - 1, total_samples)
