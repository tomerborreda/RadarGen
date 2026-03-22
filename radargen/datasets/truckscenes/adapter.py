"""TruckScenes dataset adapter implementation."""

import os.path as osp
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
from pyquaternion import Quaternion

from radargen.core.protocols import DatasetAdapter, DatasetConfig
from radargen.datasets.registry import register_adapter
from radargen.datasets.truckscenes.config import (
    TRUCKSCENES_CAMERA_VIEWS,
    TRUCKSCENES_CONFIG,
    TRUCKSCENES_RADAR_SENSORS,
    TRUCKSCENES_REFERENCE_SENSOR,
)
from radargen.datasets.truckscenes.reader import (
    get_all_scene_tokens,
    get_illuminated_scene_tokens,
    get_scene_samples,
    get_scene_samples_data,
    get_sensor_from_sensor,
    get_sensor_to_vehicle_flat_up,
    get_split_scene_tokens,
    load_radar_multisweep,
)


@register_adapter("truckscenes")
class TruckScenesAdapter(DatasetAdapter):
    """
    Dataset adapter for the MAN TruckScenes dataset.

    This adapter provides a unified interface to the TruckScenes dataset,
    handling all dataset-specific operations like loading images, computing
    transforms, and accessing radar point clouds.

    Example usage:
        >>> from truckscenes import TruckScenes
        >>> trucksc = TruckScenes("v1.0-mini", "/path/to/data")
        >>> adapter = TruckScenesAdapter(trucksc)
        >>>
        >>> # Get scene tokens for training split
        >>> scene_tokens = adapter.get_scene_tokens("train")
        >>>
        >>> # Get samples for a scene
        >>> samples = adapter.get_scene_samples(scene_tokens[0])
        >>>
        >>> # Load camera images
        >>> images = adapter.load_camera_images(samples[0])
    """

    def __init__(
        self,
        trucksc=None,
        config: Optional[DatasetConfig] = None,
        camera_views: Optional[List[str]] = None,
        reference_sensor: Optional[str] = None,
        illuminated_only: bool = True,
        dataset_dir: Optional[str] = None,
        dataset_version: Optional[str] = None,
    ):
        """
        Initialize the TruckScenes adapter.

        Args:
            trucksc: TruckScenes database object (optional if dataset_dir and dataset_version are provided)
            config: Optional custom DatasetConfig. If None, uses default.
            camera_views: Optional list of camera view names. If None, uses default.
            reference_sensor: Optional reference sensor name. If None, uses default.
            illuminated_only: If True, only use illuminated scenes (default True)
            dataset_dir: Path to dataset root directory (alternative to passing trucksc)
            dataset_version: Dataset version string (alternative to passing trucksc)
        """
        # Handle two initialization patterns:
        # 1. Pre-existing trucksc object passed directly
        # 2. dataset_dir + dataset_version provided, create trucksc internally
        if trucksc is None:
            if dataset_dir is None or dataset_version is None:
                raise ValueError(
                    "Either 'trucksc' object or both 'dataset_dir' and 'dataset_version' must be provided"
                )
            from truckscenes import TruckScenes
            trucksc = TruckScenes(version=dataset_version, dataroot=dataset_dir, verbose=False)

        self.trucksc = trucksc
        self.illuminated_only = illuminated_only

        # Build configuration
        if config is not None:
            self._config = config
        else:
            self._config = DatasetConfig(
                name="truckscenes",
                camera_views=camera_views or TRUCKSCENES_CAMERA_VIEWS,
                reference_sensor=reference_sensor or TRUCKSCENES_REFERENCE_SENSOR,
                normalization=TRUCKSCENES_CONFIG.normalization,
                data_root=trucksc.dataroot,
                version=trucksc.version,
            )

    # -------------------------------------------------------------------------
    # Iterator Methods
    # -------------------------------------------------------------------------

    def iter_samples(
        self,
        split: str,
        filter_fn: Optional[Callable[[Any], bool]] = None,
    ) -> Iterator[Any]:
        """Iterate over all samples in a split."""
        for scene_token, samples in self.iter_scenes(split):
            for sample in samples:
                if filter_fn is None or filter_fn(sample):
                    yield sample

    def iter_consecutive_pairs(
        self,
        split: str,
        filter_fn: Optional[Callable[[Any, Any], bool]] = None,
    ) -> Iterator[Tuple[Any, Any]]:
        """Iterate over consecutive sample pairs."""
        for scene_token, samples in self.iter_scenes(split):
            for i in range(len(samples) - 1):
                sample_t0 = samples[i]
                sample_t1 = samples[i + 1]
                if filter_fn is None or filter_fn(sample_t0, sample_t1):
                    yield sample_t0, sample_t1

    def iter_scenes(
        self,
        split: str,
        filter_fn: Optional[Callable[[str], bool]] = None,
    ) -> Iterator[Tuple[str, List[Any]]]:
        """Iterate over scenes with their samples."""
        scene_tokens = get_split_scene_tokens(
            self.trucksc,
            split,
            illuminated_scenes_only=self.illuminated_only,
        )
        for scene_token in scene_tokens:
            if filter_fn is None or filter_fn(scene_token):
                samples = get_scene_samples_data(
                    self.trucksc,
                    scene_token,
                    camera_freq=self._config.normalization.camera_freq,
                )
                yield scene_token, samples

    def iter_keyframes(
        self,
        split: str,
        filter_fn: Optional[Callable[[str], bool]] = None,
    ) -> Iterator[Tuple[str, List[Any]]]:
        """
        Iterate over scenes with their keyframe samples only.

        Returns only official annotated keyframes (2 Hz for TruckScenes),
        not interpolated samples at camera frequency.

        Args:
            split: Data split name ("train", "val", "test")
            filter_fn: Optional filter predicate. Called with scene_token,
                      returns True to keep scene, False to skip.

        Yields:
            Tuples of (scene_token, keyframes_list) where keyframes are
            official annotated samples
        """
        scene_tokens = get_split_scene_tokens(
            self.trucksc,
            split,
            illuminated_scenes_only=self.illuminated_only,
        )
        for scene_token in scene_tokens:
            if filter_fn is None or filter_fn(scene_token):
                keyframes = self.get_keyframe_scene_samples(scene_token)
                yield scene_token, keyframes

    def get_next_immediate_sample(self, sample_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Return the next camera-frequency sample after `sample_data`.

        Advances all cameras one step via their sample_data['next'] linked list
        and temporally matches non-camera sensors to CAMERA_FRONT's new timestamp.
        Returns None if `sample_data` is the last frame in the scene.
        """
        cam_tokens = {}
        non_cam_tokens = {}
        for key, token in sample_data.items():
            if key.startswith('_'):
                continue
            sd = self.trucksc.get('sample_data', token)
            if sd['sensor_modality'] == 'camera':
                cam_tokens[key] = token
            else:
                non_cam_tokens[key] = token

        # Advance all cameras one step; return None at end of scene
        next_cam_tokens = {}
        for key, token in cam_tokens.items():
            sd = self.trucksc.get('sample_data', token)
            if sd['next'] == '':
                return None
            next_cam_tokens[key] = sd['next']

        # Target time = CAMERA_FRONT's new timestamp
        ref_key = 'CAMERA_FRONT' if 'CAMERA_FRONT' in next_cam_tokens else next(iter(next_cam_tokens))
        target_time = self.trucksc.get('sample_data', next_cam_tokens[ref_key])['timestamp']

        # Advance non-camera sensors to the timestamp closest to target_time
        next_non_cam_tokens = {}
        for key, token in non_cam_tokens.items():
            curr_token = token
            prev_token = curr_token
            while True:
                sd = self.trucksc.get('sample_data', curr_token)
                if sd['timestamp'] >= target_time or sd['next'] == '':
                    break
                prev_token = curr_token
                curr_token = sd['next']
            curr_time = self.trucksc.get('sample_data', curr_token)['timestamp']
            prev_time = self.trucksc.get('sample_data', prev_token)['timestamp']
            next_non_cam_tokens[key] = (
                curr_token
                if abs(curr_time - target_time) <= abs(prev_time - target_time)
                else prev_token
            )

        return {**next_cam_tokens, **next_non_cam_tokens}

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def get_all_scene_tokens(self) -> List[str]:
        """
        Get all scene tokens regardless of split.

        Returns:
            List of all scene token strings
        """
        if self.illuminated_only:
            return get_illuminated_scene_tokens(self.trucksc)
        return get_all_scene_tokens(self.trucksc)

    def get_keyframe_scene_samples(self, scene_token: str) -> List[Dict[str, Any]]:
        """
        Get keyframe samples for a scene.

        This returns only the keyframe samples from TruckScenes.

        Args:
            scene_token: Scene identifier token

        Returns:
            List of keyframe sample dictionaries
        """
        return get_scene_samples(self.trucksc, scene_token)

    # -------------------------------------------------------------------------
    # Camera Data
    # -------------------------------------------------------------------------

    def get_camera_intrinsics(self, sample_data: Dict[str, Any]) -> List[np.ndarray]:
        """
        Get camera intrinsic matrices for all configured cameras.

        Args:
            sample_data: Sample data dictionary with sensor tokens

        Returns:
            List of 3x3 intrinsic matrices (one per camera view)
        """
        intrinsics = []
        for camera_view in self.camera_views:
            camera_token = sample_data[camera_view]
            camera_data = self.trucksc.get("sample_data", camera_token)
            calibrated_sensor = self.trucksc.get(
                "calibrated_sensor", camera_data["calibrated_sensor_token"]
            )
            intrinsic = np.array(calibrated_sensor["camera_intrinsic"], dtype=np.float32)
            intrinsics.append(intrinsic)
        return intrinsics

    def get_camera_extrinsics(
        self,
        sample_data: Dict[str, Any],
        ref_sensor_token: Optional[str] = None,
    ) -> List[np.ndarray]:
        """
        Get camera extrinsic matrices (camera -> reference frame).

        Args:
            sample_data: Sample data dictionary with sensor tokens
            ref_sensor_token: Optional reference sensor token

        Returns:
            List of 4x4 transformation matrices (one per camera view)
        """
        if ref_sensor_token is None:
            ref_sensor_token = sample_data.get(self.reference_sensor)
            if ref_sensor_token is None:
                raise ValueError(
                    f"ref_sensor_token must be provided if {self.reference_sensor} "
                    "is not in sample_data"
                )

        extrinsics = []
        for camera_view in self.camera_views:
            camera_token = sample_data[camera_view]
            _, _, transformation_matrix = get_sensor_from_sensor(
                self.trucksc,
                camera_token,  # source (camera)
                ref_sensor_token,  # target (reference)
            )
            extrinsics.append(transformation_matrix.astype(np.float32))
        return extrinsics

    def load_camera_images(self, sample_data: Dict[str, Any]) -> List[np.ndarray]:
        """
        Load camera images for all configured camera views.

        Args:
            sample_data: Sample data dictionary with sensor tokens

        Returns:
            List of RGB images as numpy arrays (H, W, 3)
        """
        images = []
        for camera_view in self.camera_views:
            camera_token = sample_data[camera_view]
            camera_data = self.trucksc.get("sample_data", camera_token)
            image_path = osp.join(self.trucksc.dataroot, camera_data["filename"])
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            images.append(image)
        return images

    def get_camera_image_shapes(self, sample_data: Dict[str, Any]) -> List[Tuple[int, int]]:
        """
        Get image dimensions for all configured cameras.

        Args:
            sample_data: Sample data dictionary with sensor tokens

        Returns:
            List of (height, width) tuples, one per camera view
        """
        shapes = []
        for camera_view in self.camera_views:
            camera_token = sample_data[camera_view]
            cam_sd = self.trucksc.get("sample_data", camera_token)
            shapes.append((cam_sd["height"], cam_sd["width"]))
        return shapes

    def get_camera_timestamps(self, sample_data: Dict[str, Any]) -> List[int]:
        """
        Get hardware timestamps for each camera view in microseconds.

        Returns:
            List of timestamps in microseconds, one per camera view (same order
            as self.camera_views).
        """
        timestamps = []
        for camera_view in self.camera_views:
            camera_token = sample_data[camera_view]
            camera_data = self.trucksc.get("sample_data", camera_token)
            timestamps.append(camera_data["timestamp"])
        return timestamps

    # -------------------------------------------------------------------------
    # Reference Frame Transforms
    # -------------------------------------------------------------------------

    def get_ref_to_vehicle_flat_up_transform(self, ref_sensor_token: str) -> np.ndarray:
        """
        Get transformation from reference sensor to vehicle flat-up frame.

        The flat-up frame is aligned with the ground plane (Z pointing up) and
        oriented with the vehicle's yaw direction.

        Coordinate axes in vehicle flat-up frame:
            - X-axis: Points right
            - Y-axis: Points forward (vehicle heading direction)
            - Z-axis: Points up (perpendicular to ground)

        Args:
            ref_sensor_token: Reference sensor token

        Returns:
            4x4 transformation matrix
        """
        return get_sensor_to_vehicle_flat_up(self.trucksc, ref_sensor_token).astype(
            np.float32
        )

    # -------------------------------------------------------------------------
    # Radar Data
    # -------------------------------------------------------------------------

    def load_radar_pointcloud(
        self,
        sample_data: Dict[str, Any],
        nsweeps: int = 1,
        min_distance: float = 0.0,
    ) -> np.ndarray:
        """
        Load fused radar point cloud for a sample.

        Fuses radar point clouds from all radar sensors, transforming positions
        to the vehicle flat-up frame. Velocities and radial velocity remain in
        the sensor frame (true Doppler measurement).

        Args:
            sample_data: Sample data dictionary with sensor tokens
            nsweeps: Number of radar sweeps to fuse (default 1)
            min_distance: Minimum distance from sensor to keep points (default 0.0)

        Returns:
            Point cloud array with shape (N, 8) containing:
            [x, y, z, vx, vy, vz, rcs, radial_velocity]
        """
        ref_sensor_token = sample_data.get(self.reference_sensor)
        if ref_sensor_token is None:
            raise ValueError(
                f"{self.reference_sensor} not found in sample_data"
            )

        all_pc_list = []

        for radar_sensor in TRUCKSCENES_RADAR_SENSORS:
            if radar_sensor not in sample_data:
                continue

            radar_sd_token = sample_data[radar_sensor]
            points, _ = load_radar_multisweep(
                self.trucksc,
                radar_sd_token=radar_sd_token,
                ref_sd_token=ref_sensor_token,
                nsweeps=nsweeps,
                min_distance=min_distance,
            )
            # Transform points to vehicle flat-up frame.
            def transform_points_to_vehicle_flat_up(points: np.ndarray) -> np.ndarray:
                from radargen.radar_maps.transforms import pc_transform
                transformation_matrix = get_sensor_to_vehicle_flat_up(self.trucksc, ref_sensor_token)
                # points has shape (8, N) - extract first 3 rows (x, y, z)
                # pc_transform expects (N, 3), so transpose
                points_location = points[:3, :].T  # Shape: (N, 3)
                points_location = pc_transform(points_location, transformation_matrix)  # Returns (N, 3)
                # Put transformed positions back (transpose to get (3, N))
                points[:3, :] = points_location.T
                return points
            points = transform_points_to_vehicle_flat_up(points)

            if points.shape[1] == 0:
                raise RuntimeError(f"No points found in radar sensor {radar_sensor}, this is unexpected.")
            all_pc_list.append(points.T)  # (N, 8)

        if len(all_pc_list) == 0:
            raise RuntimeError(f"Radar sensors could not be found in sample {sample_data}, this is unexpected. Check that TRUCKSCENES_RADAR_SENSORS is correctly configured.")

        return np.vstack(all_pc_list).astype(np.float32)

    # -------------------------------------------------------------------------
    # Annotations
    # -------------------------------------------------------------------------

    def get_bounding_boxes(
        self,
        sample_data: Dict[str, Any],
        ref_sensor_token: Optional[str] = None,
    ) -> List[Any]:
        """
        Get 3D bounding boxes for objects in the sample.

        Filters boxes by visibility level and spatial range (±coordinate_range).

        Args:
            sample_data: Sample data dictionary with sensor tokens and annotations information (keyframe sample)
            ref_sensor_token: Optional reference sensor token

        Returns:
            List of Box objects in ego vehicle flat-up coordinates

        Raises:
            ValueError: If sample has no annotations or ref_sensor_token is invalid
        """
        from truckscenes.utils.geometry_utils import BoxVisibility

        if ref_sensor_token is None:
            ref_sensor_token = sample_data.get(self.reference_sensor)
            if ref_sensor_token is None:
                raise ValueError(
                    f"No reference sensor token found. "
                    f"sample_data keys: {list(sample_data.keys())}"
                )
        
        ref_sample_data = self.trucksc.get('sample_data', ref_sensor_token)
        sample_token = ref_sample_data['sample_token']
        sample = self.trucksc.get('sample', sample_token)

        if "anns" not in sample:
            raise ValueError(
                f"Sample does not contain annotations information. "
                f"sample keys: {list(sample.keys())}"
            )
        if len(sample['anns']) == 0:
            return [] # No annotations

        # Filter annotations by visibility level
        # Exclude visibility level 1 (not visible) and 2 (barely visible)
        filtered_annotations = []
        for ann_token in sample["anns"]:
            ann = self.trucksc.get("sample_annotation", ann_token)
            if ann["visibility_token"] == "":
                continue
            visibility = self.trucksc.get("visibility", ann["visibility_token"])
            if visibility["level"] in [1, 2]:
                continue
            filtered_annotations.append(ann_token)

        # Get boxes in ego vehicle flat-up coordinates
        # use_flat_vehicle_coordinates=True returns boxes already transformed
        _, sample_boxes, _ = self.trucksc.get_sample_data(
            ref_sensor_token,
            box_vis_level=BoxVisibility.ANY,
            use_flat_vehicle_coordinates=True,
            selected_anntokens=filtered_annotations,
        )

        # Filter boxes by spatial range
        coordinate_range = self._config.normalization.coordinate_range
        filtered_boxes = []
        for box in sample_boxes:
            center = box.center
            if abs(center[0]) <= coordinate_range and abs(center[1]) <= coordinate_range:
                filtered_boxes.append(box)

        return filtered_boxes

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_camera_image_paths(self, sample_data: Dict[str, Any]) -> List[str]:
        """
        Get file paths for camera images.

        Args:
            sample_data: Sample data dictionary

        Returns:
            List of image file paths
        """
        paths = []
        for camera_view in self.camera_views:
            camera_token = sample_data[camera_view]
            camera_data = self.trucksc.get("sample_data", camera_token)
            paths.append(osp.join(self.trucksc.dataroot, camera_data["filename"]))
        return paths

    def get_num_frames_in_scene(self, scene_token: str) -> int:
        """
        Get the number of frames in a scene.

        Args:
            scene_token: Scene identifier token

        Returns:
            Number of frames in the scene
        """
        samples = self.get_scene_samples(scene_token)
        return len(samples)

    # -------------------------------------------------------------------------
    # Filter Helper Methods
    # -------------------------------------------------------------------------

    def has_radar_data(self, sample: Any) -> bool:
        """
        Check if sample has radar data.

        Args:
            sample: Sample object (dict with sensor tokens)

        Returns:
            True if sample has at least one radar sensor, False otherwise
        """
        if not isinstance(sample, dict):
            return False
        return any(radar in sample for radar in TRUCKSCENES_RADAR_SENSORS)

    def has_annotations(self, sample: Any) -> bool:
        """
        Check if sample has annotations.

        For TruckScenes, annotations are on official keyframe samples.
        Interpolated samples from get_scene_samples_data do not have annotations.

        Args:
            sample: Sample object (dict)

        Returns:
            True if sample has annotations, False otherwise
        """
        if not isinstance(sample, dict):
            return False
        # Check if this is an official annotated sample
        # Official samples have 'anns' key or are referenced by 'token'
        return 'anns' in sample or 'token' in sample
