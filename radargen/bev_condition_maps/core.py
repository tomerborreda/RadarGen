"""Core BEV creation logic shared across inference and preprocessing pipelines."""

from typing import Callable, List, Tuple, Union

import numpy as np
import torch

from radargen.bev_condition_maps.foundation_models import (
    get_depth,
    get_depth_batched,
    segment_image,
    segment_images_batched,
    radial_velocity_from_flow,
    radial_velocity_from_flow_batched,
    get_points_mask,
    points_to_bev_map,
)


def create_bev_maps_from_camera_data(
    camera_images_t0: List[np.ndarray],
    camera_images_t1: List[np.ndarray],
    camera_intrinsics: List[np.ndarray],
    transform_fn: Callable[[List[dict]], np.ndarray],
    models: dict,
    device: torch.device,
    dts: List[float],
    resolution: int,
    coordinate_range: float,
    doppler_min: float,
    doppler_max: float,
    use_batched_inference: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create BEV conditioning maps from raw camera data.

    Low-level function that processes multi-view camera images through foundation
    models and converts to BEV representation. This is the core BEV creation logic
    shared between inference and preprocessing pipelines.

    Args:
        camera_images_t0: List of RGB images at time t0, shape (H, W, 3) each
        camera_images_t1: List of RGB images at time t1, shape (H, W, 3) each
        camera_intrinsics: List of 3x3 camera intrinsic matrices
        transform_fn: Function to transform depth outputs to BEV reference frame.
                     Signature: (depth_outputs: List[dict]) -> np.ndarray of shape (N, 3)
        models: Dict with foundation models:
                - 'depth': UniDepthV2 model
                - 'segmentation': Mask2Former model
                - 'segmentation_processor': Mask2Former processor
                - 'flow': UniFlowMatch model
        device: PyTorch device
        dts: Per-camera time deltas in seconds (one value per camera view)
        resolution: Output BEV map resolution (e.g., 512)
        coordinate_range: Spatial range in meters (e.g., 50.0)
        doppler_min: Minimum radial velocity bound in m/s
        doppler_max: Maximum radial velocity bound in m/s
        use_batched_inference: If True, use batched model inference for speedup

    Returns:
        Tuple of (bev_color_map, bev_seg_map, bev_velocity_map)
        - bev_color_map: (resolution, resolution, 3) RGB appearance map
        - bev_seg_map: (resolution, resolution, 3) Segmentation map
        - bev_velocity_map: (resolution, resolution, 3) Radial velocity map
    """
    # 1. Run foundation models on all cameras
    if use_batched_inference:
        # Check if all cameras share the same intrinsics for potential batching
        all_intrinsics_same = all(
            np.allclose(camera_intrinsics[0], K, rtol=1e-5, atol=1e-8)
            for K in camera_intrinsics[1:]
        ) if len(camera_intrinsics) > 1 else True

        # Depth inference - batch if all intrinsics are the same, otherwise sequential
        if all_intrinsics_same:
            # All cameras have same intrinsics - can use batched depth inference
            depth_outputs_t0 = get_depth_batched(
                models['depth'], camera_images_t0, camera_intrinsics, device
            )
            depth_outputs_t1 = get_depth_batched(
                models['depth'], camera_images_t1, camera_intrinsics, device
            )
        else:
            # Different intrinsics - fall back to sequential processing
            # Print warning only once (check if we've already warned)
            if not hasattr(create_bev_maps_from_camera_data, '_intrinsics_warning_shown'):
                print("⚠️  Cameras have different intrinsics - using sequential depth inference")
                print("   (Batched segmentation still enabled for speedup)")
                create_bev_maps_from_camera_data._intrinsics_warning_shown = True

            depth_outputs_t0 = [
                get_depth(models['depth'], img, intrinsics, device)
                for img, intrinsics in zip(camera_images_t0, camera_intrinsics)
            ]
            depth_outputs_t1 = [
                get_depth(models['depth'], img, intrinsics, device)
                for img, intrinsics in zip(camera_images_t1, camera_intrinsics)
            ]

        # Batched segmentation (works well with multiple images)
        segmented_images = segment_images_batched(
            models['segmentation'], models['segmentation_processor'],
            camera_images_t0, device
        )

        # Flow - sequential (UFM processes pairs individually)
        flow_results = radial_velocity_from_flow_batched(
            models['flow'], camera_images_t0, depth_outputs_t0,
            camera_images_t1, depth_outputs_t1,
            device, dts=dts,
            doppler_min=doppler_min, doppler_max=doppler_max,
        )
        velocity_images = [result[1] for result in flow_results]
    else:
        # Original sequential processing (for backwards compatibility)
        depth_outputs_t0 = [
            get_depth(models['depth'], img, intrinsics, device)
            for img, intrinsics in zip(camera_images_t0, camera_intrinsics)
        ]
        depth_outputs_t1 = [
            get_depth(models['depth'], img, intrinsics, device)
            for img, intrinsics in zip(camera_images_t1, camera_intrinsics)
        ]

        segmented_images = [
            segment_image(models['segmentation'], models['segmentation_processor'], img, device)
            for img in camera_images_t0
        ]

        velocity_images = [
            radial_velocity_from_flow(
                models['flow'], img_t0, depth_t0, img_t1, depth_t1,
                device, dt=dt,
                doppler_min=doppler_min, doppler_max=doppler_max,
            )[1]
            for img_t0, depth_t0, img_t1, depth_t1, dt in zip(
                camera_images_t0, depth_outputs_t0, camera_images_t1, depth_outputs_t1, dts
            )
        ]

    # 2. Transform depth points to BEV reference frame
    all_points = transform_fn(depth_outputs_t0)

    # 3. Get point mask (remove edges, sky)
    all_masks = get_points_mask(depth_outputs_t0, segmented_images)

    # 4. Generate BEV maps
    original_height, original_width = depth_outputs_t0[0]['points'].shape[:2]

    bev_color_map = points_to_bev_map(
        all_points, all_masks, camera_images_t0,
        original_height, original_width, resolution, coordinate_range
    )
    bev_seg_map = points_to_bev_map(
        all_points, all_masks, segmented_images,
        original_height, original_width, resolution, coordinate_range
    )
    bev_velocity_map = points_to_bev_map(
        all_points, all_masks, velocity_images,
        original_height, original_width, resolution, coordinate_range
    )

    return bev_color_map, bev_seg_map, bev_velocity_map
