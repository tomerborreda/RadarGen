"""Data types used across RadarGen for inference and training."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
import torch


@dataclass
class RadarGenInferenceInput:
    """
    Input data structure for RadarGen inference.

    This is a dataset-agnostic representation of the input data required
    for the RadarGen inference pipeline.

    Attributes:
        input_images: List of camera images as numpy arrays (H, W, 3) in RGB format.
        camera_intrinsics: List of 3x3 camera intrinsic matrices.
        camera_extrinsics: List of 4x4 transformation matrices (camera -> reference frame).
        ref_to_ego_flat_up: 4x4 transformation matrix (reference -> ego flat-up frame).
    """
    input_images: List[np.ndarray]
    camera_intrinsics: List[np.ndarray]
    camera_extrinsics: List[np.ndarray]
    ref_to_ego_flat_up: np.ndarray

    def __post_init__(self):
        """Validate input data."""
        n_cameras = len(self.input_images)
        if len(self.camera_intrinsics) != n_cameras:
            raise ValueError(
                f"Number of intrinsics ({len(self.camera_intrinsics)}) "
                f"must match number of images ({n_cameras})"
            )
        if len(self.camera_extrinsics) != n_cameras:
            raise ValueError(
                f"Number of extrinsics ({len(self.camera_extrinsics)}) "
                f"must match number of images ({n_cameras})"
            )

        for i, intrinsic in enumerate(self.camera_intrinsics):
            if intrinsic.shape != (3, 3):
                raise ValueError(f"Intrinsic {i} must be 3x3, got {intrinsic.shape}")

        for i, extrinsic in enumerate(self.camera_extrinsics):
            if extrinsic.shape != (4, 4):
                raise ValueError(f"Extrinsic {i} must be 4x4, got {extrinsic.shape}")

        if self.ref_to_ego_flat_up.shape != (4, 4):
            raise ValueError(
                f"ref_to_ego_flat_up must be 4x4, got {self.ref_to_ego_flat_up.shape}"
            )


class TrainingBatch(TypedDict, total=False):
    """
    Training batch structure with named fields.

    This replaces the previous tuple-based batch format where fields were
    accessed by magic indices (e.g., batch[8], batch[9]).

    Required fields:
        pd_map: Point density map tensor (B, C, H, W)
        rcs_map: RCS values map tensor (B, C, H, W)
        doppler_map: Doppler velocity map tensor (B, C, H, W)
        bev_color_map: BEV appearance/color map tensor (B, C, H, W)
        bev_seg_map: BEV segmentation map tensor (B, C, H, W)
        bev_velocity_map: BEV velocity map tensor (B, C, H, W)
        data_info: Dictionary with metadata (img_hw, aspect_ratio, etc.)

    Optional fields:
        scene_token: Scene identifier token
        sample_token: Sample identifier token
        ref_sensor_token: Reference sensor token
    """
    # Required fields - radar target maps
    pd_map: torch.Tensor
    rcs_map: torch.Tensor
    doppler_map: torch.Tensor

    # Required fields - BEV conditioning maps
    bev_color_map: torch.Tensor
    bev_seg_map: torch.Tensor
    bev_velocity_map: torch.Tensor

    # Required fields - metadata
    data_info: Dict[str, Any]

    # Optional fields - identifiers
    scene_token: Optional[str]
    sample_token: Optional[str]
    ref_sensor_token: Optional[str]


@dataclass
class SampleInfo:
    """
    Information about a single sample in the dataset.

    Used to track sample metadata for training and evaluation.
    """
    scene_token: str
    sample_token: str
    ref_sensor_token: str
    timestamp: Optional[int] = None
    frame_idx: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)
