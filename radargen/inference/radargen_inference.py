"""
RadarGen inference pipeline using the adapter pattern.

This module provides a dataset-agnostic inference pipeline that can work
with any DatasetAdapter implementation.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyrallis
import torch
from torch import nn

from diffusion.data.transforms import get_transform
from radargen.bev_condition_maps.core import create_bev_maps_from_camera_data
from radargen.core.data_types import RadarGenInferenceInput
from radargen.core.protocols import DatasetAdapter
from radargen.datasets.base import build_inference_input
from radargen.inference.diffusion_pipeline import RadarGenDiffusionPipeline, RadarGenInferenceArgs
from radargen.bev_condition_maps.foundation_models import load_models
from radargen.inference.pcl_recovery import recover_pcl_attributes
from radargen.radar_maps.transforms import pc_transform


class RadarGenInference(nn.Module):
    """
    Dataset-agnostic RadarGen inference pipeline.

    This class orchestrates the 3-stage RadarGen pipeline:
    1. Multi-view images -> BEV conditioning maps
    2. BEV maps -> Radar maps (using diffusion model)
    3. Radar maps -> Point cloud (using sparse deconvolution)

    Can be used with any DatasetAdapter for automatic data extraction,
    or directly with RadarGenInferenceInput objects.

    Example usage with adapter:
        >>> from truckscenes import TruckScenes
        >>> from radargen.datasets.truckscenes import TruckScenesAdapter
        >>>
        >>> trucksc = TruckScenes("v1.0-mini", "/path/to/data")
        >>> adapter = TruckScenesAdapter(trucksc)
        >>> model = RadarGenInference(adapter=adapter)
        >>>
        >>> # Iterate over consecutive pairs
        >>> for sample_t0, sample_t1 in adapter.iter_consecutive_pairs('val'):
        >>>     pcl = model.from_sample_data(sample_t0, sample_t1)
        >>>     break  # Just one example

    Example usage without adapter:
        >>> model = RadarGenInference()
        >>> input_t0 = RadarGenInferenceInput(images, intrinsics, extrinsics, ref_transform)
        >>> input_t1 = RadarGenInferenceInput(...)
        >>> pcl = model(input_t0, input_t1)
    """

    def __init__(
        self,
        config: Optional[str] = "configs/RadarGen_600M_512px_TS_inference.yaml",
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        adapter: Optional[DatasetAdapter] = None,
        use_batched_inference: bool = True,
    ):
        """
        Initialize RadarGen inference pipeline.

        Args:
            config: Path to model configuration file
            device: PyTorch device to use
            generator: Random number generator for reproducibility
            adapter: Optional DatasetAdapter for automatic data extraction
            use_batched_inference: Use batched inference for BEV creation (3-5x speedup)
        """
        super().__init__()

        self.config = pyrallis.load(RadarGenInferenceArgs, open(config))
        self.device = (
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if device is None
            else device
        )
        self.adapter = adapter

        # Get normalization from adapter or fall back to config
        extra = self.config.data.extra
        if adapter is not None:
            self.coordinate_range = adapter.normalization.coordinate_range
            self.rcs_min = adapter.normalization.rcs_min
            self.rcs_max = adapter.normalization.rcs_max
            self.doppler_min = adapter.normalization.doppler_min
            self.doppler_max = adapter.normalization.doppler_max
        else:
            self.coordinate_range = extra.get("coordinate_range", 50.0)
            self.rcs_min = extra['rcs_min']
            self.rcs_max = extra['rcs_max']
            self.doppler_min = extra['doppler_min']
            self.doppler_max = extra['doppler_max']

        # Foundational Models for BEV generation
        (
            self.depth_model,
            self.segmentation_model,
            self.segmentation_processor,
            self.flow_model,
        ) = load_models(self.device)

        # RadarGen Diffusion Model
        self.diffusion_model = RadarGenDiffusionPipeline(config=config, device=self.device)
        # Override normalization from adapter if provided
        if adapter is not None:
            self.diffusion_model.rcs_min = self.rcs_min
            self.diffusion_model.rcs_max = self.rcs_max
            self.diffusion_model.doppler_min = self.doppler_min
            self.diffusion_model.doppler_max = self.doppler_max
        self.diffusion_model.from_pretrained(self.config.model.load_from)
        self.diffusion_model.eval()

        self.generator = (
            torch.Generator(device=self.device).manual_seed(42)
            if generator is None
            else generator
        )
        self.image_size = self.config.data.image_size
        self.bev_maps_transform = get_transform("default_train_from_np", resolution=self.image_size)

        # Optimization settings
        self.use_batched_inference = use_batched_inference

    def transform_points_to_reference_frame(
        self,
        input_t0: RadarGenInferenceInput,
        depth_outputs: List[dict],
    ) -> np.ndarray:
        """
        Transform 3D points from each camera's local frame to the ego-vehicle frame.

        Args:
            input_t0: Inference input with camera transforms
            depth_outputs: List of depth output dicts with 'points' tensor

        Returns:
            Concatenated points from all cameras in ego frame, shape (N, 3)
        """
        all_points = []
        for depth_output, extrinsic in zip(depth_outputs, input_t0.camera_extrinsics):
            # Get points from depth output: (H, W, 3) -> (H*W, 3)
            points = depth_output["points"].reshape(-1, 3).cpu().numpy()

            # Transform from camera frame to reference frame
            points_ref = pc_transform(points, extrinsic)

            # Transform from reference frame to ego flat-up frame
            points_ego = pc_transform(points_ref, input_t0.ref_to_ego_flat_up)

            all_points.append(points_ego)

        return np.concatenate(all_points, axis=0)

    def mv_to_bev_maps(
        self,
        input_t0: RadarGenInferenceInput,
        input_t1: RadarGenInferenceInput,
        dts: List[float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert multi-view camera images to BEV conditioning maps.

        Stage 1 of the RadarGen pipeline: Uses foundation models (UniDepthV2,
        Mask2Former, UniFlowMatch) to generate BEV appearance, segmentation,
        and velocity maps.

        Args:
            input_t0: Camera input at time t
            input_t1: Camera input at time t+1 (for velocity estimation)
            dts: Per-camera time deltas in seconds (one per camera view)

        Returns:
            Tuple of (bev_appearance_map, bev_segmentation_map, bev_velocity_map)
        """
        # Build models dict for shared function
        models = {
            'depth': self.depth_model,
            'segmentation': self.segmentation_model,
            'segmentation_processor': self.segmentation_processor,
            'flow': self.flow_model,
        }

        # Create transformation function that uses this instance's method
        def transform_fn(depth_outputs: List[dict]) -> np.ndarray:
            return self.transform_points_to_reference_frame(input_t0, depth_outputs)

        # Generate BEV maps using shared logic
        return create_bev_maps_from_camera_data(
            camera_images_t0=input_t0.input_images,
            camera_images_t1=input_t1.input_images,
            camera_intrinsics=input_t0.camera_intrinsics,
            transform_fn=transform_fn,
            models=models,
            device=self.device,
            dts=dts,
            resolution=self.image_size,
            coordinate_range=self.coordinate_range,
            use_batched_inference=self.use_batched_inference,
            doppler_min=self.doppler_min,
            doppler_max=self.doppler_max,
        )

    def bev_maps_to_radar_maps(
        self,
        bev_appearance_map: np.ndarray,
        bev_segmentation_map: np.ndarray,
        bev_velocity_map: np.ndarray,
        num_inference_steps: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert BEV conditioning maps to radar maps using diffusion.

        Stage 2 of the RadarGen pipeline: Uses a diffusion model
        to generate Point Density, RCS, and Doppler maps.

        Args:
            bev_appearance_map: BEV color/appearance map
            bev_segmentation_map: BEV semantic segmentation map
            bev_velocity_map: BEV velocity map
            num_inference_steps: Number of diffusion denoising steps

        Returns:
            Tuple of (pd_map, rcs_map, doppler_map)
        """
        bev_color_map = self.bev_maps_transform(bev_appearance_map).to(self.device).unsqueeze(0)
        bev_seg_map = self.bev_maps_transform(bev_segmentation_map).to(self.device).unsqueeze(0)
        bev_vel_map = self.bev_maps_transform(bev_velocity_map).to(self.device).unsqueeze(0)

        with torch.no_grad():
            generated_maps = self.diffusion_model(
                prompt="",
                height_single_image=self.image_size,
                width_single_image=self.image_size,
                height_num_images=1,
                width_num_images=1,
                guidance_scale=0.0,
                pag_guidance_scale=0.0,
                num_inference_steps=num_inference_steps,
                generator=self.generator,
                bev_condition_maps=[bev_color_map, bev_seg_map, bev_vel_map],
            )[0]

        pd_map_np = generated_maps.pd_map_np
        rcs_map_np = generated_maps.rcs_map_np
        doppler_map_np = generated_maps.doppler_map_np

        return pd_map_np, rcs_map_np, doppler_map_np

    def radar_maps_to_pcl(
        self,
        pd_map: np.ndarray,
        rcs_map: np.ndarray,
        doppler_map: np.ndarray,
        z_value: float = 0.0,
    ) -> np.ndarray:
        """
        Convert radar maps to a point cloud using sparse deconvolution.

        Stage 3 of the RadarGen pipeline: Uses IRL1 + FISTA to recover
        sparse point cloud from the generated radar maps.

        Args:
            pd_map: Point density map (H, W)
            rcs_map: RCS values map (H, W)
            doppler_map: Doppler values map (H, W)
            z_value: Z coordinate for all points (default 0.0)

        Returns:
            Point cloud array with shape (N, 5) containing [x, y, z, rcs, doppler]
        """
        image_size = pd_map.shape[0]

        recovered_locations, recovered_rcs_values, recovered_doppler_values = (
            recover_pcl_attributes(
                pd_map=pd_map,
                rcs_map=rcs_map,
                doppler_map=doppler_map,
                image_size=image_size,
                coordinate_range=self.coordinate_range,
                device=self.device,
            )
        )

        num_points_recovered = len(recovered_locations)
        assert num_points_recovered > 0, "No points recovered"
        
        # Create point cloud: [x, y, z, rcs, doppler]
        z_coords = np.full((num_points_recovered, 1), z_value, dtype=np.float32)
        points = np.concatenate(
            [
                recovered_locations,  # x, y
                z_coords,  # z
                recovered_rcs_values[:, None],  # rcs
                recovered_doppler_values[:, None],  # doppler
            ],
            axis=1,
        )

        return points.astype(np.float32)

    def get_pcl(
        self,
        input_t0: RadarGenInferenceInput,
        input_t1: RadarGenInferenceInput,
        num_inference_steps: int = 20,
        dts: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Run the full RadarGen pipeline to generate a radar point cloud.

        Args:
            input_t0: Camera input at time t
            input_t1: Camera input at time t+1
            num_inference_steps: Number of diffusion denoising steps
            dts: Per-camera time deltas in seconds (one per camera view)

        Returns:
            Point cloud array with shape (N, 5) containing [x, y, z, rcs, doppler]
        """
        # 1. BEV Scene Conditioning
        bev_appearance_map, bev_segmentation_map, bev_velocity_map = self.mv_to_bev_maps(
            input_t0, input_t1, dts=dts
        )

        # 2. Conditional Radar Maps Denoising
        pd_map, rcs_map, doppler_map = self.bev_maps_to_radar_maps(
            bev_appearance_map, bev_segmentation_map, bev_velocity_map, num_inference_steps
        )

        # 3. Recovering Radar PCL
        pcl = self.radar_maps_to_pcl(pd_map, rcs_map, doppler_map)

        return pcl

    def get_pcl_from_bev_maps(
        self,
        bev_appearance_map: np.ndarray,
        bev_segmentation_map: np.ndarray,
        bev_velocity_map: np.ndarray,
        num_inference_steps: int = 20,
    ) -> np.ndarray:
        """
        Run RadarGen pipeline starting from pre-computed BEV conditioning maps.

        This method skips stage 1 (BEV generation from camera images) and runs
        only stages 2-3:
        - Stage 2: Conditional radar map denoising (diffusion model)
        - Stage 3: Sparse deconvolution to recover point cloud (IRL1 + FISTA)

        Use this when BEV maps are pre-computed (e.g., during evaluation) to
        significantly speed up inference by avoiding the foundation model forward
        passes (UniDepth, Mask2Former, UniFlowMatch).

        Args:
            bev_appearance_map: RGB appearance map, shape (H, W, 3), dtype uint8
            bev_segmentation_map: Segmentation map, shape (H, W, 3), dtype uint8
            bev_velocity_map: Velocity field, shape (H, W, 1), dtype float32
            num_inference_steps: Number of diffusion denoising steps

        Returns:
            Point cloud array with shape (N, 5) containing [x, y, z, rcs, doppler]

        Example:
            >>> # Load pre-computed BEV maps
            >>> bev_color, bev_seg, bev_vel = adapter.load_bev_condition_maps(
            ...     scene_token="abc123", frame_idx=0, bev_maps_dir="maps/bev"
            ... )
            >>> # Run inference starting from stage 2
            >>> pcl = model.get_pcl_from_bev_maps(bev_color, bev_seg, bev_vel)
        """
        # Stage 2: Conditional Radar Maps Denoising
        pd_map, rcs_map, doppler_map = self.bev_maps_to_radar_maps(
            bev_appearance_map, bev_segmentation_map, bev_velocity_map, num_inference_steps
        )

        # Stage 3: Recovering Radar PCL
        pcl = self.radar_maps_to_pcl(pd_map, rcs_map, doppler_map)

        return pcl

    def from_sample_data(
        self,
        sample_data_t0: Dict[str, Any],
        sample_data_t1: Dict[str, Any],
        ref_sensor_token: Optional[str] = None,
        num_inference_steps: int = 20,
    ) -> np.ndarray:
        """
        Run inference directly from sample data using the adapter.

        This is a convenience method when using the adapter pattern.

        Args:
            sample_data_t0: Sample data dictionary for time t
            sample_data_t1: Sample data dictionary for time t+1
            ref_sensor_token: Optional reference sensor token
            num_inference_steps: Number of diffusion denoising steps

        Returns:
            Point cloud array with shape (N, 5) containing [x, y, z, rcs, doppler]

        Raises:
            ValueError: If no adapter is configured
        """
        if self.adapter is None:
            raise ValueError(
                "No adapter configured. Either pass an adapter to __init__ or use "
                "get_pcl() with RadarGenInferenceInput objects directly."
            )

        input_t0 = build_inference_input(
            self.adapter, sample_data_t0, ref_sensor_token
        )
        input_t1 = build_inference_input(
            self.adapter, sample_data_t1, ref_sensor_token
        )

        timestamps_t0 = self.adapter.get_camera_timestamps(sample_data_t0)
        timestamps_t1 = self.adapter.get_camera_timestamps(sample_data_t1)
        dts = [abs(ts1 - ts0) / 1e6 for ts0, ts1 in zip(timestamps_t0, timestamps_t1)]

        return self.get_pcl(input_t0, input_t1, num_inference_steps, dts=dts)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a checkpoint for the diffusion model.

        This method allows loading a checkpoint after initialization, which is
        useful for evaluation scenarios where you want to override the default
        checkpoint path from the config file.

        Args:
            checkpoint_path: Path or HuggingFace model ID to load
        """
        self.diffusion_model.from_pretrained(checkpoint_path)

    def forward(
        self,
        input_t0: RadarGenInferenceInput,
        input_t1: RadarGenInferenceInput,
        num_inference_steps: int = 20,
        dts: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Forward pass - same as get_pcl()."""
        return self.get_pcl(input_t0, input_t1, num_inference_steps, dts=dts)
