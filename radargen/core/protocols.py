"""Protocol definitions for dataset adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

from radargen.core.normalization import NormalizationConfig, default_normalization


@dataclass
class DatasetConfig:
    """
    Configuration for a dataset adapter.

    This dataclass holds all dataset-specific configuration including
    camera views, sensor references, and normalization parameters.

    Attributes:
        name: Dataset name (e.g., "truckscenes", "nuscenes")
        camera_views: List of camera view names to use
        reference_sensor: Name of the reference sensor for coordinate transforms
        normalization: Normalization configuration for this dataset
        data_root: Root directory of the dataset
        version: Dataset version string
        extra: Additional dataset-specific configuration
    """
    name: str
    camera_views: List[str]
    reference_sensor: str
    normalization: NormalizationConfig = field(default_factory=lambda: default_normalization)
    data_root: Optional[str] = None
    version: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class DatasetAdapter(ABC):
    """
    Abstract base class for dataset adapters.

    Dataset adapters provide a unified interface for accessing data from
    different autonomous driving datasets (TruckScenes, nuScenes, etc.).
    Each adapter handles the dataset-specific logic for loading images,
    computing transforms, and accessing annotations.

    Implementations should:
    1. Initialize with dataset-specific parameters
    2. Set the _config attribute with a DatasetConfig
    3. Implement all abstract methods
    """

    _config: DatasetConfig

    @property
    def config(self) -> DatasetConfig:
        """Return the dataset configuration."""
        return self._config

    @property
    def normalization(self) -> NormalizationConfig:
        """Return the normalization configuration."""
        return self._config.normalization

    @property
    def camera_views(self) -> List[str]:
        """Return the list of camera view names."""
        return self._config.camera_views

    @property
    def reference_sensor(self) -> str:
        """Return the reference sensor name."""
        return self._config.reference_sensor

    # -------------------------------------------------------------------------
    # Iterator Methods
    # -------------------------------------------------------------------------

    @abstractmethod
    def iter_samples(
        self,
        split: str,
        filter_fn: Optional[Callable[[Any], bool]] = None,
    ) -> Iterator[Any]:
        """
        Iterate over all samples in a split.

        Args:
            split: Data split name ("train", "val", "test")
            filter_fn: Optional filter predicate. Called with sample object,
                      returns True to keep, False to skip.

        Yields:
            Sample objects (adapter-specific structure)

        Example:
            >>> for sample in adapter.iter_samples('train'):
            ...     images = adapter.load_camera_images(sample)

            >>> # With filtering
            >>> for sample in adapter.iter_samples('val', filter_fn=adapter.has_all_sensors):
            ...     pcl = adapter.load_radar_pointcloud(sample)
        """
        pass

    @abstractmethod
    def iter_consecutive_pairs(
        self,
        split: str,
        filter_fn: Optional[Callable[[Any, Any], bool]] = None,
    ) -> Iterator[Tuple[Any, Any]]:
        """
        Iterate over consecutive sample pairs for velocity estimation.

        Automatically excludes last sample of each scene (no t+1 available).

        Args:
            split: Data split name
            filter_fn: Optional filter predicate. Called with (sample_t0, sample_t1),
                      returns True to keep pair, False to skip.

        Yields:
            Tuples of (sample_t0, sample_t1)

        Example:
            >>> for sample_t0, sample_t1 in adapter.iter_consecutive_pairs('train'):
            ...     bev_maps = generate_bev_maps(adapter, sample_t0, sample_t1)
        """
        pass

    @abstractmethod
    def iter_scenes(
        self,
        split: str,
        filter_fn: Optional[Callable[[str], bool]] = None,
    ) -> Iterator[Tuple[str, List[Any]]]:
        """
        Iterate over scenes with their samples.

        Useful for scene-level processing and multi-GPU distribution.

        Args:
            split: Data split name
            filter_fn: Optional filter predicate. Called with scene_token,
                      returns True to keep scene, False to skip.

        Yields:
            Tuples of (scene_token, samples_list)

        Example:
            >>> # Multi-GPU distribution
            >>> scenes = list(adapter.iter_scenes('train'))
            >>> local_scenes = scenes[rank::world_size]
            >>> for scene_token, samples in local_scenes:
            ...     process_scene(scene_token, samples)
        """
        pass

    @abstractmethod
    def iter_keyframes(
        self,
        split: str,
        filter_fn: Optional[Callable[[str], bool]] = None,
    ) -> Iterator[Tuple[str, List[Any]]]:
        """
        Iterate over scenes with their keyframe samples only.

        Keyframes are the official annotated samples in a dataset, as opposed to
        intermediate/interpolated samples. This is useful for evaluation where
        you only want to evaluate on ground-truth annotated frames.

        Args:
            split: Data split name
            filter_fn: Optional filter predicate. Called with scene_token,
                      returns True to keep scene, False to skip.

        Yields:
            Tuples of (scene_token, keyframe_samples_list)

        Example:
            >>> # Iterate only over annotated keyframes for evaluation
            >>> for scene_token, keyframes in adapter.iter_keyframes('val'):
            ...     for sample in keyframes:
            ...         evaluate_sample(sample)
        """
        pass

    def iter_keyframes_indexed(
        self, split: str
    ) -> Iterator[Tuple[str, List[Tuple[Any, int]]]]:
        """
        Iterate over keyframe samples paired with their camera-frequency frame index.

        The index matches the ordering of adapter.iter_scenes() and is the indexing
        convention used in precomputed BEV map filenames:
            bev_*_{scene_token}_{frame_idx}.npy

        Uses a dual-pointer walk: advances through the camera-frequency list from
        iter_scenes() while stepping through the keyframe list from iter_keyframes(),
        matching frames by the first camera view's token. Camera tokens are the
        most reliable signal because both iter_scenes() and iter_keyframes() walk
        the same camera sample_data linked list from the same starting point.
        Matching by reference sensor (lidar/radar) is unreliable because
        get_scene_all_samples() may snap non-camera sensors to slightly different
        tokens depending on synthetic vs actual timestamps.

        Yields:
            (scene_token, [(keyframe_sample, camera_freq_idx), ...])
            Only indices with a valid t+1 pair are included (last sample excluded).
        """
        first_cam = self.camera_views[0]
        all_scenes: Dict[str, List[Any]] = dict(self.iter_scenes(split))

        for scene_token, keyframe_samples in self.iter_keyframes(split):
            all_samples = all_scenes[scene_token]

            kf_iter = iter(keyframe_samples)
            next_kf = next(kf_iter, None)

            indexed: List[Tuple[Any, int]] = []
            for idx, sample in enumerate(all_samples[:-1]):  # exclude last (no t+1)
                if next_kf is None:
                    break
                sd = sample["data"] if isinstance(sample, dict) and "data" in sample else sample
                kf_sd = next_kf["data"] if isinstance(next_kf, dict) and "data" in next_kf else next_kf
                if sd.get(first_cam) == kf_sd.get(first_cam):
                    indexed.append((next_kf, idx))
                    next_kf = next(kf_iter, None)

            yield scene_token, indexed

    @abstractmethod
    def get_next_immediate_sample(self, sample_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Return the next camera-frequency sample after `sample_data`.

        Unlike using the next entry from iter_keyframes(), this follows the
        full-frequency camera linked list one step forward (~0.1s), not the
        next keyframe (~0.5s). Non-camera sensors are temporally matched to
        the new camera timestamp.

        This is needed for correct velocity/Doppler map calculation when
        evaluating in keyframes_only mode.

        Args:
            sample_data: Flat sensor-token dict (same format returned by
                         iter_scenes / iter_keyframes after unwrapping 'data').

        Returns:
            A new sample_data dict at the next camera-frequency step, or None
            if `sample_data` is the last frame in the scene.
        """
        pass

    # -------------------------------------------------------------------------
    # Camera Data
    # -------------------------------------------------------------------------

    @abstractmethod
    def get_camera_intrinsics(self, sample_data: Dict[str, Any]) -> List[np.ndarray]:
        """
        Get camera intrinsic matrices for all configured cameras.

        Args:
            sample_data: Sample data dictionary with sensor tokens

        Returns:
            List of 3x3 intrinsic matrices (one per camera view)
        """
        pass

    @abstractmethod
    def get_camera_extrinsics(
        self,
        sample_data: Dict[str, Any],
        ref_sensor_token: Optional[str] = None,
    ) -> List[np.ndarray]:
        """
        Get camera extrinsic matrices (camera -> reference frame).

        Args:
            sample_data: Sample data dictionary with sensor tokens
            ref_sensor_token: Optional reference sensor token. If None, uses
                            the default reference sensor.

        Returns:
            List of 4x4 transformation matrices (one per camera view)
        """
        pass

    @abstractmethod
    def load_camera_images(self, sample_data: Dict[str, Any]) -> List[np.ndarray]:
        """
        Load camera images for all configured cameras.

        Args:
            sample_data: Sample data dictionary with sensor tokens

        Returns:
            List of RGB images as numpy arrays (H, W, 3)
        """
        pass

    @abstractmethod
    def get_camera_image_shapes(self, sample_data: Dict[str, Any]) -> List[Tuple[int, int]]:
        """
        Get image dimensions for all configured cameras.

        Args:
            sample_data: Sample data dictionary with sensor tokens

        Returns:
            List of (height, width) tuples, one per camera view
        """
        pass

    @abstractmethod
    def get_camera_timestamps(self, sample_data: Dict[str, Any]) -> List[int]:
        """
        Get hardware timestamps for each camera view in microseconds.

        Returns:
            List of timestamps in microseconds, one per camera view (same order
            as self.camera_views).
        """
        pass

    # -------------------------------------------------------------------------
    # Reference Frame Transforms
    # -------------------------------------------------------------------------

    @abstractmethod
    def get_ref_to_vehicle_flat_up_transform(self, ref_sensor_token: str) -> np.ndarray:
        """
        Get transformation from reference sensor to vehicle flat-up frame.

        The flat-up frame is aligned with the ground plane (Z pointing up) and
        oriented with the vehicle's yaw direction. This is the standard BEV frame.

        Args:
            ref_sensor_token: Reference sensor token

        Returns:
            4x4 transformation matrix
        """
        pass

    # -------------------------------------------------------------------------
    # Radar Data
    # -------------------------------------------------------------------------

    @abstractmethod
    def load_radar_pointcloud(
        self,
        sample_data: Dict[str, Any],
        nsweeps: int = 1,
        min_distance: float = 0.0,
    ) -> np.ndarray:
        """
        Load fused radar point cloud for a sample.

        The reference sensor token is resolved internally from sample_data
        using the adapter's configured reference_sensor.

        Args:
            sample_data: Sample data dictionary with sensor tokens
            nsweeps: Number of radar sweeps to accumulate (default=1)
            min_distance: Minimum distance from sensor to keep points (default 0.0)

        Returns:
            Point cloud array with shape (N, 8) where columns are:
            [x, y, z, vx, vy, vz, rcs, radial_velocity]
            - x, y, z: Position in vehicle flat-up frame (meters). The flat-up
                       frame is ground-plane aligned with Z pointing up,
                       oriented with the vehicle's yaw direction (standard BEV frame).
            - vx, vy, vz: Velocity in sensor frame (m/s)
            - rcs: Radar Cross Section (dBsm)
            - radial_velocity: Radial velocity component (m/s)
        """
        pass

    # -------------------------------------------------------------------------
    # Pre-computed BEV Condition Maps
    # -------------------------------------------------------------------------

    def load_bev_condition_maps(
        self,
        scene_token: str,
        frame_idx: int,
        bev_maps_dir: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load pre-computed BEV conditioning maps for a sample.

        Loads three BEV maps using the standard naming convention:
        - bev_color_map_{scene_token}_{frame_idx}.npy
        - bev_seg_map_{scene_token}_{frame_idx}.npy
        - bev_velocity_map_{scene_token}_{frame_idx}.npy

        This is a concrete method (not abstract) that works for all datasets,
        as BEV maps are computed from camera images using foundation models
        independent of the dataset structure. The naming convention matches
        the output format from scripts/create_bev_condition_maps.py.

        Subclasses can override this method if they use a different storage
        format or naming convention.

        Args:
            scene_token: Scene identifier token
            frame_idx: 0-based frame index within the scene
            bev_maps_dir: Directory containing pre-computed BEV maps

        Returns:
            Tuple of (bev_color_map, bev_seg_map, bev_velocity_map)
            - bev_color_map: np.ndarray shape (H, W, 3), RGB appearance, dtype uint8
            - bev_seg_map: np.ndarray shape (H, W, 3), segmentation, dtype uint8
            - bev_velocity_map: np.ndarray shape (H, W, 1), velocity field, dtype float32

        Raises:
            FileNotFoundError: If any required map file is missing
        """
        from pathlib import Path

        bev_maps_dir = Path(bev_maps_dir)

        bev_color_map = np.load(
            bev_maps_dir / f"bev_color_map_{scene_token}_{frame_idx}.npy"
        )
        bev_seg_map = np.load(
            bev_maps_dir / f"bev_seg_map_{scene_token}_{frame_idx}.npy"
        )
        bev_velocity_map = np.load(
            bev_maps_dir / f"bev_velocity_map_{scene_token}_{frame_idx}.npy"
        )

        return bev_color_map, bev_seg_map, bev_velocity_map

    # -------------------------------------------------------------------------
    # Annotations
    # -------------------------------------------------------------------------

    @abstractmethod
    def get_bounding_boxes(
        self,
        sample_data: Dict[str, Any],
        ref_sensor_token: Optional[str] = None,
    ) -> List[Any]:
        """
        Get 3D bounding boxes for objects in the sample.

        Args:
            sample_data: Sample data dictionary with sensor tokens
            ref_sensor_token: Optional reference sensor token for coordinate transform

        Returns:
            List of Box objects (or equivalent) for detected objects
        """
        pass

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def filter_by_camera_frustum(
        self,
        pointcloud: np.ndarray,
        sample_data: Dict[str, Any],
        camera_views: Optional[List[str]] = None,
        ref_to_points_transform: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Filter point cloud to keep only points visible in at least one camera.

        Uses camera intrinsics, extrinsics, and image shapes to determine
        which points project into at least one camera's field of view.

        Args:
            pointcloud: Point cloud array (N, 8+) with x,y,z in first 3 columns
            sample_data: Sample data dictionary with sensor tokens
            camera_views: Optional subset of camera views to filter by.
                         If None, uses all configured camera views.
            ref_to_points_transform: Optional 4x4 transformation matrix from
                                    reference frame to point cloud frame.
                                    If None, assumes points are in reference frame.
                                    Example: pass ref_to_flat_up for points in flat-up frame.

        Returns:
            Filtered point cloud with same column structure. Shape: (M, 8+) where M <= N.
        """
        from radargen.core.geometry import view_points

        if pointcloud.shape[0] == 0:
            return pointcloud

        views = camera_views if camera_views is not None else self.camera_views

        intrinsics = self.get_camera_intrinsics(sample_data)
        extrinsics = self.get_camera_extrinsics(sample_data)
        image_shapes = self.get_camera_image_shapes(sample_data)

        positions = pointcloud[:, :3]  # (N, 3)
        frustum_mask = np.zeros(pointcloud.shape[0], dtype=bool)

        for i, view_name in enumerate(self.camera_views):
            if view_name not in views:
                continue

            camera_intrinsic = intrinsics[i]
            # Extrinsics are camera→reference.
            cam_to_ref = extrinsics[i]

            # If points are in a different frame, transform camera extrinsics to that frame
            if ref_to_points_transform is not None:
                # cam_to_points = ref_to_points @ cam_to_ref
                cam_to_points = ref_to_points_transform @ cam_to_ref
                # Now invert to get points→camera
                R = cam_to_points[:3, :3]
                t = cam_to_points[:3, 3]
                points_to_cam = np.eye(4)
                points_to_cam[:3, :3] = R.T
                points_to_cam[:3, 3] = -R.T @ t
            else:
                # Points are in reference frame, invert to get ref→camera
                R = cam_to_ref[:3, :3]
                t = cam_to_ref[:3, 3]
                points_to_cam = np.eye(4)
                points_to_cam[:3, :3] = R.T
                points_to_cam[:3, 3] = -R.T @ t

            # Transform points to camera frame
            points_3d = positions.T  # (3, N)
            points_cam = points_to_cam.dot(
                np.vstack((points_3d, np.ones(points_3d.shape[1])))
            )[:3, :]

            # Project to 2D
            points_2d = view_points(points_cam, camera_intrinsic, normalize=True)

            # Check visibility
            img_h, img_w = image_shapes[i]
            in_front = points_cam[2, :] > 0
            in_image_x = (points_2d[0, :] >= 0) & (points_2d[0, :] < img_w)
            in_image_y = (points_2d[1, :] >= 0) & (points_2d[1, :] < img_h)

            frustum_mask |= in_front & in_image_x & in_image_y

        return pointcloud[frustum_mask].copy()

    def get_reference_sensor_token(self, sample_data: Dict[str, Any]) -> str:
        """
        Get the reference sensor token from sample data.

        Args:
            sample_data: Sample data dictionary

        Returns:
            Reference sensor token string
        """
        return sample_data[self.reference_sensor]

    # -------------------------------------------------------------------------
    # Filter Helper Methods
    # -------------------------------------------------------------------------

    def has_all_sensors(self, sample: Any) -> bool:
        """
        Check if sample has all required sensors (cameras + reference).

        Args:
            sample: Sample object (adapter-specific structure)

        Returns:
            True if sample has all required sensors, False otherwise
        """
        # Assumes sample is dict-like, but subclasses can override
        if not isinstance(sample, dict):
            return True  # Adapter-specific sample object, assume valid
        for camera in self.camera_views:
            if camera not in sample:
                return False
        return self.reference_sensor in sample

    @abstractmethod
    def has_radar_data(self, sample: Any) -> bool:
        """
        Check if sample has radar sensor data (dataset-specific).

        Args:
            sample: Sample object (adapter-specific structure)

        Returns:
            True if sample has radar data, False otherwise
        """
        pass

    @abstractmethod
    def has_annotations(self, sample: Any) -> bool:
        """
        Check if sample has bounding box annotations (dataset-specific).

        Args:
            sample: Sample object (adapter-specific structure)

        Returns:
            True if sample has annotations, False otherwise
        """
        pass
