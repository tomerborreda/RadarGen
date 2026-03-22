"""nuScenes dataset adapter implementation."""

import os.path as osp
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
from nuscenes import NuScenes
from pyquaternion import Quaternion

from radargen.core.protocols import DatasetAdapter, DatasetConfig
from radargen.datasets.nuscenes.config import (
    NUSCENES_CAMERA_VIEWS,
    NUSCENES_CONFIG,
    NUSCENES_RADAR_SENSORS,
    NUSCENES_REFERENCE_SENSOR,
)
from radargen.datasets.nuscenes.reader import (
    get_scene_all_samples,
    get_scene_keyframe_samples,
    get_sensor_to_reference_transform,
    get_sensor_to_vehicle_flat_up,
    get_split_scene_tokens,
    load_radar_multisweep,
)
from radargen.datasets.registry import register_adapter


@register_adapter("nuscenes")
class NuScenesAdapter(DatasetAdapter):
    """
    Dataset adapter for the nuScenes autonomous driving dataset.

    This adapter provides a unified interface to the nuScenes dataset,
    handling all dataset-specific operations like loading images, computing
    transforms, and accessing radar point clouds.

    nuScenes has 6 surround-view cameras, 5 radar sensors, and 1 lidar.
    Keyframes are annotated at ~2 Hz; all camera frames are at ~12 Hz.

    Example usage:
        >>> from nuscenes import NuScenes
        >>> nusc = NuScenes("v1.0-mini", "/path/to/data")
        >>> adapter = NuScenesAdapter(nusc)
        >>>
        >>> # Get keyframe samples for a scene
        >>> for scene_token, keyframes in adapter.iter_keyframes("mini_train"):
        ...     imgs = adapter.load_camera_images(keyframes[0])   # 6 images
        ...     pcl = adapter.load_radar_pointcloud(keyframes[0]) # (N, 8)
    """

    def __init__(
        self,
        nusc=None,
        config: Optional[DatasetConfig] = None,
        camera_views: Optional[List[str]] = None,
        reference_sensor: Optional[str] = None,
        dataset_dir: Optional[str] = None,
        dataset_version: Optional[str] = None,
    ):
        """
        Initialize the nuScenes adapter.

        Args:
            nusc: NuScenes database object (optional if dataset_dir and
                  dataset_version are provided)
            config: Optional custom DatasetConfig. If None, uses default.
            camera_views: Optional list of camera view names. If None, uses default.
            reference_sensor: Optional reference sensor name. If None, uses default.
            dataset_dir: Path to dataset root directory (alternative to nusc)
            dataset_version: Dataset version string (alternative to nusc)
        """
        if nusc is None:
            if dataset_dir is None or dataset_version is None:
                raise ValueError(
                    "Either 'nusc' object or both 'dataset_dir' and "
                    "'dataset_version' must be provided"
                )
            nusc = NuScenes(version=dataset_version, dataroot=dataset_dir, verbose=False)

        self.nusc = nusc

        if config is not None:
            self._config = config
        else:
            self._config = DatasetConfig(
                name="nuscenes",
                camera_views=camera_views or NUSCENES_CAMERA_VIEWS,
                reference_sensor=reference_sensor or NUSCENES_REFERENCE_SENSOR,
                normalization=NUSCENES_CONFIG.normalization,
                data_root=nusc.dataroot,
                version=nusc.version,
            )

    # -------------------------------------------------------------------------
    # Iterator Methods
    # -------------------------------------------------------------------------

    def iter_samples(
        self,
        split: str,
        filter_fn: Optional[Callable[[Any], bool]] = None,
    ) -> Iterator[Any]:
        """Iterate over all samples at camera frequency (~12 Hz) in a split."""
        for scene_token, samples in self.iter_scenes(split):
            for sample in samples:
                if filter_fn is None or filter_fn(sample):
                    yield sample

    def iter_consecutive_pairs(
        self,
        split: str,
        filter_fn: Optional[Callable[[Any, Any], bool]] = None,
    ) -> Iterator[Tuple[Any, Any]]:
        """Iterate over consecutive sample pairs at camera frequency."""
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
        """Iterate over scenes with all camera-frequency samples (~12 Hz)."""
        scene_tokens = get_split_scene_tokens(self.nusc, split)
        for scene_token in scene_tokens:
            if filter_fn is None or filter_fn(scene_token):
                samples = get_scene_all_samples(
                    self.nusc,
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
        Iterate over scenes with their keyframe samples only (~2 Hz).

        Uses the nuScenes native sample linked list (sample['next']) which
        directly gives annotated keyframes with all sensor data available.

        Args:
            split: Data split name ("train", "val", "mini_train", "mini_val", etc.)
            filter_fn: Optional filter predicate. Called with scene_token,
                      returns True to keep scene, False to skip.

        Yields:
            Tuples of (scene_token, keyframes_list)
        """
        scene_tokens = get_split_scene_tokens(self.nusc, split)
        for scene_token in scene_tokens:
            if filter_fn is None or filter_fn(scene_token):
                keyframes = get_scene_keyframe_samples(self.nusc, scene_token)
                yield scene_token, keyframes

    def get_next_immediate_sample(self, sample_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Return the next camera-frequency sample after `sample_data`.

        Advances all cameras one step via their sample_data['next'] linked list
        and temporally matches non-camera sensors to CAM_FRONT's new timestamp.
        Returns None if `sample_data` is the last frame in the scene.
        """
        cam_tokens = {}
        non_cam_tokens = {}
        for key, token in sample_data.items():
            if key.startswith('_'):
                continue
            sd = self.nusc.get('sample_data', token)
            if sd['sensor_modality'] == 'camera':
                cam_tokens[key] = token
            else:
                non_cam_tokens[key] = token

        # Advance all cameras one step; return None at end of scene
        next_cam_tokens = {}
        for key, token in cam_tokens.items():
            sd = self.nusc.get('sample_data', token)
            if sd['next'] == '':
                return None
            next_cam_tokens[key] = sd['next']

        # Target time = CAM_FRONT's new timestamp
        ref_key = 'CAM_FRONT' if 'CAM_FRONT' in next_cam_tokens else next(iter(next_cam_tokens))
        target_time = self.nusc.get('sample_data', next_cam_tokens[ref_key])['timestamp']

        # Advance non-camera sensors to the timestamp closest to target_time
        next_non_cam_tokens = {}
        for key, token in non_cam_tokens.items():
            curr_token = token
            prev_token = curr_token
            while True:
                sd = self.nusc.get('sample_data', curr_token)
                if sd['timestamp'] >= target_time or sd['next'] == '':
                    break
                prev_token = curr_token
                curr_token = sd['next']
            curr_time = self.nusc.get('sample_data', curr_token)['timestamp']
            prev_time = self.nusc.get('sample_data', prev_token)['timestamp']
            next_non_cam_tokens[key] = (
                curr_token
                if abs(curr_time - target_time) <= abs(prev_time - target_time)
                else prev_token
            )

        return {
            **next_cam_tokens,
            **next_non_cam_tokens,
            '_timestamp': target_time,
            '_sample_token': sample_data.get('_sample_token', ''),
        }

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def get_scene_tokens(self, split: str) -> List[str]:
        """
        Get scene tokens for a given split.

        Args:
            split: Data split name

        Returns:
            List of scene token strings
        """
        return get_split_scene_tokens(self.nusc, split)

    def get_scene_samples(self, scene_token: str) -> List[Dict[str, Any]]:
        """
        Get keyframe samples for a scene.

        Args:
            scene_token: Scene identifier token

        Returns:
            List of keyframe sample dicts
        """
        return get_scene_keyframe_samples(self.nusc, scene_token)

    def get_keyframe_scene_samples(self, scene_token: str) -> List[Dict[str, Any]]:
        """
        Get keyframe samples for a scene (same as get_scene_samples).

        Args:
            scene_token: Scene identifier token

        Returns:
            List of keyframe sample dicts
        """
        return get_scene_keyframe_samples(self.nusc, scene_token)

    # -------------------------------------------------------------------------
    # Camera Data
    # -------------------------------------------------------------------------

    def get_camera_intrinsics(self, sample_data: Dict[str, Any]) -> List[np.ndarray]:
        """
        Get camera intrinsic matrices for all configured cameras.

        Args:
            sample_data: Sample data dict with sensor tokens

        Returns:
            List of 3x3 intrinsic matrices (one per camera view)
        """
        intrinsics = []
        for camera_view in self.camera_views:
            camera_token = sample_data[camera_view]
            camera_sd = self.nusc.get("sample_data", camera_token)
            cs = self.nusc.get(
                "calibrated_sensor", camera_sd["calibrated_sensor_token"]
            )
            intrinsic = np.array(cs["camera_intrinsic"], dtype=np.float32)
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
            sample_data: Sample data dict with sensor tokens
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
            transformation_matrix = get_sensor_to_reference_transform(
                self.nusc,
                source_sd_token=camera_token,
                target_sd_token=ref_sensor_token,
            )
            extrinsics.append(transformation_matrix.astype(np.float32))
        return extrinsics

    def load_camera_images(self, sample_data: Dict[str, Any]) -> List[np.ndarray]:
        """
        Load camera images for all configured camera views.

        Args:
            sample_data: Sample data dict with sensor tokens

        Returns:
            List of RGB images as numpy arrays (H, W, 3)
        """
        images = []
        for camera_view in self.camera_views:
            camera_token = sample_data[camera_view]
            camera_sd = self.nusc.get("sample_data", camera_token)
            image_path = osp.join(self.nusc.dataroot, camera_sd["filename"])
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            images.append(image)
        return images

    def get_camera_image_shapes(
        self, sample_data: Dict[str, Any]
    ) -> List[Tuple[int, int]]:
        """
        Get image dimensions for all configured cameras.

        Args:
            sample_data: Sample data dict with sensor tokens

        Returns:
            List of (height, width) tuples, one per camera view
        """
        shapes = []
        for camera_view in self.camera_views:
            camera_token = sample_data[camera_view]
            cam_sd = self.nusc.get("sample_data", camera_token)
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
            camera_sd = self.nusc.get("sample_data", camera_token)
            timestamps.append(camera_sd["timestamp"])
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
            ref_sensor_token: Reference sensor sample_data token

        Returns:
            4x4 transformation matrix
        """
        return get_sensor_to_vehicle_flat_up(self.nusc, ref_sensor_token).astype(
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

        Fuses radar point clouds from all 5 radar sensors, transforming
        positions to the vehicle flat-up frame. Velocities (vx, vy raw) and
        radial velocity remain in sensor frame (true Doppler measurement).

        Args:
            sample_data: Sample data dict with sensor tokens
            nsweeps: Number of radar sweeps to fuse (default 1)
            min_distance: Minimum distance from sensor to keep points (default 0.0)

        Returns:
            Point cloud array (N, 8):
            [x, y, z, vx, vy, vz, rcs, radial_velocity]
            in vehicle flat-up frame (positions), sensor frame (velocities)
        """
        ref_sensor_token = sample_data.get(self.reference_sensor)
        if ref_sensor_token is None:
            raise ValueError(
                f"{self.reference_sensor} not found in sample_data"
            )

        all_pc_list = []

        for radar_sensor in NUSCENES_RADAR_SENSORS:
            if radar_sensor not in sample_data:
                continue

            radar_sd_token = sample_data[radar_sensor]
            points, _ = load_radar_multisweep(
                self.nusc,
                radar_sd_token=radar_sd_token,
                ref_sd_token=ref_sensor_token,
                nsweeps=nsweeps,
                min_distance=min_distance,
            )

            # Transform positions to vehicle flat-up frame
            if points.shape[1] > 0:
                transformation_matrix = get_sensor_to_vehicle_flat_up(
                    self.nusc, ref_sensor_token
                )
                n = points.shape[1]
                pos_hom = np.vstack([points[:3, :], np.ones((1, n))])
                points[:3, :] = (transformation_matrix @ pos_hom)[:3, :]

                all_pc_list.append(points.T)  # (N, 8)

        if len(all_pc_list) == 0:
            return np.zeros((0, 8), dtype=np.float32)

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

        Returns boxes in vehicle flat-up coordinates. Filters by visibility
        (excludes levels 1 and 2) and spatial range (±coordinate_range).

        Args:
            sample_data: Sample data dict with sensor tokens and '_sample_token'
            ref_sensor_token: Optional reference sensor token

        Returns:
            List of Box objects in ego vehicle flat-up coordinates
        """
        if ref_sensor_token is None:
            ref_sensor_token = sample_data.get(self.reference_sensor)
            if ref_sensor_token is None:
                raise ValueError(
                    f"No reference sensor token found. "
                    f"sample_data keys: {list(sample_data.keys())}"
                )

        # Get the sample token (from keyframe dict or from sample_data record)
        sample_token = sample_data.get("_sample_token")
        if sample_token is None:
            ref_sd = self.nusc.get("sample_data", ref_sensor_token)
            sample_token = ref_sd["sample_token"]

        sample = self.nusc.get("sample", sample_token)

        if "anns" not in sample or len(sample["anns"]) == 0:
            return []

        # Filter by visibility: exclude levels 1 (not visible) and 2 (barely visible)
        filtered_anntokens = []
        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)
            if ann["visibility_token"] == "":
                continue
            visibility = self.nusc.get("visibility", ann["visibility_token"])
            if int(visibility["token"]) in [1, 2]:
                continue
            filtered_anntokens.append(ann_token)

        # Get boxes transformed to ego vehicle flat-up frame
        transformation_matrix = get_sensor_to_vehicle_flat_up(
            self.nusc, ref_sensor_token
        )

        boxes = []
        for ann_token in filtered_anntokens:
            ann = self.nusc.get("sample_annotation", ann_token)
            # nuScenes Box in global frame, transform to flat-up
            from nuscenes.utils.data_classes import Box
            box = Box(
                center=ann["translation"],
                size=ann["size"],
                orientation=Quaternion(ann["rotation"]),
                name=ann["category_name"],
                token=ann["token"],
            )
            # Transform: global -> ref sensor frame
            ref_sd = self.nusc.get("sample_data", ref_sensor_token)
            ref_pose = self.nusc.get("ego_pose", ref_sd["ego_pose_token"])
            ref_cs = self.nusc.get("calibrated_sensor", ref_sd["calibrated_sensor_token"])

            # global -> ego
            box.translate(-np.array(ref_pose["translation"]))
            box.rotate(Quaternion(ref_pose["rotation"]).inverse)
            # ego -> ref sensor
            box.translate(-np.array(ref_cs["translation"]))
            box.rotate(Quaternion(ref_cs["rotation"]).inverse)
            # ref sensor -> flat-up
            box.rotate(Quaternion(matrix=transformation_matrix[:3, :3]))
            box.translate(transformation_matrix[:3, 3])

            # Filter by spatial range
            coordinate_range = self._config.normalization.coordinate_range
            center = box.center
            if (
                abs(center[0]) <= coordinate_range
                and abs(center[1]) <= coordinate_range
            ):
                boxes.append(box)

        return boxes

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_camera_image_paths(self, sample_data: Dict[str, Any]) -> List[str]:
        """
        Get file paths for camera images.

        Args:
            sample_data: Sample data dict

        Returns:
            List of image file paths
        """
        paths = []
        for camera_view in self.camera_views:
            camera_token = sample_data[camera_view]
            camera_sd = self.nusc.get("sample_data", camera_token)
            paths.append(osp.join(self.nusc.dataroot, camera_sd["filename"]))
        return paths

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
        return any(radar in sample for radar in NUSCENES_RADAR_SENSORS)

    def has_annotations(self, sample: Any) -> bool:
        """
        Check if sample has annotations.

        For nuScenes, annotations exist on official keyframe samples.
        Keyframes have '_sample_token' key set by get_scene_keyframe_samples().

        Args:
            sample: Sample object (dict)

        Returns:
            True if sample has annotations, False otherwise
        """
        if not isinstance(sample, dict):
            return False
        return "_sample_token" in sample
