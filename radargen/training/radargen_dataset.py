"""
RadarGen training dataset using the adapter pattern.

This module provides a dataset-agnostic training dataset that works with any
DatasetAdapter implementation (TruckScenes, nuScenes, etc.).
"""

import os
import os.path as osp
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from diffusion.data.transforms import get_transform
from diffusion.utils.logger import get_root_logger
from radargen.core.data_types import TrainingBatch
from radargen.core.protocols import DatasetAdapter
from radargen.radar_maps import doppler_map_to_image, pd_map_to_image, rcs_map_to_image


class RadarGenDataset(Dataset):
    """
    Dataset-agnostic training dataset for RadarGen.

    This dataset uses a DatasetAdapter to load data from various autonomous
    driving datasets. It returns TrainingBatch dictionaries with named fields
    instead of tuples with magic indices.

    The dataset loads:
    - Radar maps (Point Density, RCS, Doppler) as target outputs
    - BEV conditioning maps (appearance, segmentation, velocity) as inputs

    Args:
        adapter: DatasetAdapter instance for the dataset
        split: Data split to use ("train", "val", "test")
        resolution: Image resolution for maps (default 512)
        radar_maps_dir: Directory containing pre-computed radar maps
        bev_conditioning_maps_dir: Directory containing pre-computed BEV maps
        transform: Optional transform to apply to images/maps
        config: Optional config object with additional settings

    Example usage:
        >>> from radargen.datasets.truckscenes import TruckScenesAdapter
        >>> from truckscenes import TruckScenes
        >>>
        >>> trucksc = TruckScenes("v1.0-trainval", "/data/truckscenes")
        >>> adapter = TruckScenesAdapter(trucksc)
        >>> dataset = RadarGenDataset(
        ...     adapter=adapter,
        ...     split="train",
        ...     radar_maps_dir="radar_maps",
        ...     bev_conditioning_maps_dir="bev_conditioning_maps",
        ... )
        >>> batch = dataset[0]
        >>> print(batch["pd_map"].shape)  # Use named keys
    """

    def __init__(
        self,
        adapter: DatasetAdapter,
        split: str,
        resolution: int = 512,
        radar_maps_dir: str = "radar_maps",
        bev_conditioning_maps_dir: str = "bev_conditioning_maps",
        transform=None,
        map_transform=None,
        config: Optional[Any] = None,
        logger=None,
    ):
        self.adapter = adapter
        self.split = split
        self.resolution = resolution
        self.radar_maps_dir = radar_maps_dir
        self.bev_conditioning_maps_dir = bev_conditioning_maps_dir
        self.map_transform = map_transform
        self.config = config
        self.logger = logger

        # Build frame index - list of (scene_token, frame_idx, sample_data) tuples
        self._build_frame_index()

        # For compatibility with diffusion/data/builder.py logging
        self.ori_imgs_nums = len(self.frame_index)

        if self.logger:
            self.logger.info(f"RadarGenDataset: {len(self)} samples for split '{split}'")

    def _build_frame_index(self):
        """
        Build index of all frames across scenes.

        Note: We exclude the last sample of each scene because BEV velocity maps
        require consecutive frame pairs (t, t+1). The last sample has no t+1.

        This method demonstrates both iteration patterns:
        - NEW: Using iter_scenes() with manual slice (cleaner)
        - OLD: Manual get_scene_tokens() + get_scene_samples() (commented)

        Both produce identical results. The new pattern is slightly cleaner.
        """
        self.frame_index: List[Tuple[str, int, Dict[str, Any]]] = []

        # NEW PATTERN (RECOMMENDED): Using iter_scenes()
        # Cleaner iteration with less boilerplate
        for scene_token, samples in self.adapter.iter_scenes(self.split):
            # Remove last sample for velocity maps (need t and t+1)
            samples = samples[:-1]
            for frame_idx, sample_data in enumerate(samples):
                self.frame_index.append((scene_token, frame_idx, sample_data))

        # OLD PATTERN (FOR REFERENCE - EQUIVALENT TO ABOVE):
        # scene_tokens = self.adapter.get_scene_tokens(self.split)
        # for scene_token in scene_tokens:
        #     samples = self.adapter.get_scene_samples(scene_token)
        #     # Remove last sample for velocity maps (need t and t+1)
        #     samples = samples[:-1]
        #     for frame_idx, sample_data in enumerate(samples):
        #         self.frame_index.append((scene_token, frame_idx, sample_data))

        # ALTERNATIVE (MOST CONCISE): Using iter_consecutive_pairs()
        # If you only need sample_t0 and don't care about scene_token/frame_idx:
        # self.frame_index = [
        #     (None, i, sample_t0)
        #     for i, (sample_t0, _) in enumerate(self.adapter.iter_consecutive_pairs(self.split))
        # ]

    def __len__(self) -> int:
        return len(self.frame_index)

    def _load_radar_maps(
        self,
        scene_token: str,
        frame_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load pre-computed radar maps for a sample.

        Args:
            scene_token: Scene identifier token
            frame_idx: Frame index within scene

        Returns:
            Tuple of (pd_map, rcs_map, doppler_map) as numpy arrays
        """
        # Load Point Density Map
        pd_map_path = osp.join(
            self.radar_maps_dir, f"radar_pd_map_{scene_token}_{frame_idx}.npy"
        )
        pd_map = np.load(pd_map_path)
        _, pd_map_np_image = pd_map_to_image(pd_map)

        # Load RCS Map
        rcs_map_path = osp.join(
            self.radar_maps_dir, f"radar_rcs_map_{scene_token}_{frame_idx}.npy"
        )
        rcs_map = np.load(rcs_map_path)
        norm = self.adapter.normalization
        _, rcs_map_np_image = rcs_map_to_image(rcs_map, rcs_min=norm.rcs_min, rcs_max=norm.rcs_max)

        # Load Doppler Map
        doppler_map_path = osp.join(
            self.radar_maps_dir, f"radar_doppler_map_{scene_token}_{frame_idx}.npy"
        )
        doppler_map = np.load(doppler_map_path)
        _, doppler_map_np_image = doppler_map_to_image(doppler_map, doppler_min=norm.doppler_min, doppler_max=norm.doppler_max)

        return pd_map_np_image, rcs_map_np_image, doppler_map_np_image

    def _load_bev_maps(
        self,
        scene_token: str,
        frame_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load pre-computed BEV conditioning maps for a sample.

        Uses the adapter's load_bev_condition_maps method for consistent
        loading across training and evaluation.

        Args:
            scene_token: Scene identifier token
            frame_idx: Frame index within scene

        Returns:
            Tuple of (color_map, seg_map, velocity_map) as numpy arrays
        """
        return self.adapter.load_bev_condition_maps(
            scene_token=scene_token,
            frame_idx=frame_idx,
            bev_maps_dir=self.bev_conditioning_maps_dir,
        )

    def _apply_transform(self, np_image: np.ndarray) -> torch.Tensor:
        """Apply map transform to numpy image."""
        if self.map_transform is not None:
            return self.map_transform(np_image)
        # Default: convert to tensor and normalize to [-1, 1]
        tensor = torch.from_numpy(np_image).float()
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(-1).repeat(1, 1, 3)
        tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
        tensor = tensor / 127.5 - 1.0  # [0, 255] -> [-1, 1]
        return tensor

    def __getitem__(self, idx: int) -> TrainingBatch:
        """
        Get a training batch by index.

        Returns a TrainingBatch TypedDict with named fields instead of
        a tuple with magic indices.

        Args:
            idx: Dataset index

        Returns:
            TrainingBatch dictionary with:
            - pd_map: Point density map tensor
            - rcs_map: RCS map tensor
            - doppler_map: Doppler map tensor
            - bev_color_map: BEV appearance map tensor
            - bev_seg_map: BEV segmentation map tensor
            - bev_velocity_map: BEV velocity map tensor
            - data_info: Metadata dictionary
        """
        scene_token, frame_idx, sample_data = self.frame_index[idx]

        # Load radar target maps
        pd_map_np, rcs_map_np, doppler_map_np = self._load_radar_maps(scene_token, frame_idx)

        # Load BEV conditioning maps
        bev_color_np, bev_seg_np, bev_vel_np = self._load_bev_maps(scene_token, frame_idx)

        # Apply transforms
        pd_map_tensor = self._apply_transform(pd_map_np)
        rcs_map_tensor = self._apply_transform(rcs_map_np)
        doppler_map_tensor = self._apply_transform(doppler_map_np)
        bev_color_map_tensor = self._apply_transform(bev_color_np)
        bev_seg_map_tensor = self._apply_transform(bev_seg_np)
        bev_vel_map_tensor = self._apply_transform(bev_vel_np)

        # Build metadata
        ref_sensor_token = sample_data[self.adapter.reference_sensor]
        data_info = {
            "img_hw": torch.tensor(
                [self.resolution, self.resolution], dtype=torch.float32
            ),
            "aspect_ratio": torch.tensor(1.0),
            "scene_token": scene_token,
            "ref_sensor_token": ref_sensor_token,
            "frame_idx": frame_idx,
        }

        # Return TrainingBatch with named fields
        return {
            "pd_map": pd_map_tensor,
            "rcs_map": rcs_map_tensor,
            "doppler_map": doppler_map_tensor,
            "bev_color_map": bev_color_map_tensor,
            "bev_seg_map": bev_seg_map_tensor,
            "bev_velocity_map": bev_vel_map_tensor,
            "data_info": data_info,
        }


def build_radargen_dataset_from_config(config, **kwargs):
    """
    Build RadarGenDataset from a config object.

    This provides compatibility with the existing diffusion/data/builder.py
    registration system.
    """

    # Determine dataset type from config
    # extra is a dict when loaded from YAML
    extra = getattr(config.data, 'extra', None) or {}
    dataset_name = extra.get("dataset_name", "") if isinstance(extra, dict) else getattr(extra, 'dataset_name', "")
    if not dataset_name:
        raise ValueError("dataset_name must be specified in config.data.extra")
    dataset_dir = config.data.dataset_dir

    if dataset_name == "truckscenes":
        from truckscenes import TruckScenes

        from radargen.datasets.truckscenes import TruckScenesAdapter

        trucksc_version = extra.get("dataset_version", "v1.0-trainval")
        trucksc = TruckScenes(trucksc_version, dataset_dir, verbose=False)
        adapter = TruckScenesAdapter(trucksc)
    elif dataset_name == "nuscenes":
        from nuscenes import NuScenes

        from radargen.datasets.nuscenes import NuScenesAdapter

        nuscenes_version = extra.get("dataset_version", "v1.0-trainval")
        nusc = NuScenes(version=nuscenes_version, dataroot=dataset_dir, verbose=False)
        adapter = NuScenesAdapter(nusc)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    split = extra.get("eval_split", "train")
    resolution = config.data.image_size

    # Get transforms
    map_transform = get_transform("default_train_from_np", resolution)

    # Get logger
    logger = get_root_logger() if config is None else get_root_logger(
        osp.join(config.work_dir, "train_log.log")
    )

    # Get map directories from config
    if hasattr(config.data, 'radar_maps_dir') and hasattr(config.data, 'bev_conditioning_maps_dir'):
        radar_maps_dir = config.data.radar_maps_dir
        bev_conditioning_maps_dir = config.data.bev_conditioning_maps_dir
    else:
        raise ValueError("radar_maps_dir and bev_conditioning_maps_dir must be specified")

    # Convert relative paths to absolute by prepending dataset_dir
    if not osp.isabs(radar_maps_dir):
        radar_maps_dir = osp.join(dataset_dir, radar_maps_dir)
    if not osp.isabs(bev_conditioning_maps_dir):
        bev_conditioning_maps_dir = osp.join(dataset_dir, bev_conditioning_maps_dir)

    return RadarGenDataset(
        adapter=adapter,
        split=split,
        resolution=resolution,
        radar_maps_dir=radar_maps_dir,
        bev_conditioning_maps_dir=bev_conditioning_maps_dir,
        map_transform=map_transform,
        config=config,
        logger=logger,
    )
