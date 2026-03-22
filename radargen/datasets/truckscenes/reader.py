"""
TruckScenes data reading utilities.
"""

import os.path as osp
from functools import reduce
from typing import Any, Dict, List, Tuple

import numpy as np
from pyquaternion import Quaternion
from truckscenes.utils.data_classes import RadarPointCloud
from truckscenes.utils.geometry_utils import transform_matrix
from truckscenes.utils.splits import create_splits_scenes

from radargen.datasets.truckscenes.config import EXCLUDED_SCENE_INDEX


def get_all_scene_tokens(trucksc) -> List[str]:
    """
    Returns all scene tokens in the dataset.

    Args:
        trucksc: TruckScenes database object

    Returns:
        List of scene token strings
    """
    scene_tokens_all = [s["token"] for s in trucksc.scene]
    if len(scene_tokens_all) == 0:
        raise RuntimeError("Error: Database has no samples!")

    # Remove scene_token in index 345 because it's missing a camera token
    if trucksc.version == "v1.0-trainval":
        scene_tokens_all.pop(EXCLUDED_SCENE_INDEX)

    return scene_tokens_all


def get_illuminated_scene_tokens(trucksc) -> List[str]:
    """
    Returns the scene tokens for the illuminated scenes.

    Args:
        trucksc: TruckScenes database object

    Returns:
        List of illuminated scene token strings
    """
    scene_tokens = []
    for scene_token in get_all_scene_tokens(trucksc):
        scene_record = trucksc.get("scene", scene_token)
        if "illuminated" in scene_record["description"]:
            scene_tokens.append(scene_token)
    return scene_tokens


def get_split_scene_tokens(
    trucksc,
    eval_split: str,
    illuminated_scenes_only: bool = False,
) -> List[str]:
    """
    Returns the scene tokens for the given split.

    Args:
        trucksc: TruckScenes database object
        eval_split: Split name (e.g., "train", "val", "test")
        illuminated_scenes_only: If True, only return illuminated scenes

    Returns:
        List of scene token strings for the split
    """
    splits = create_splits_scenes()
    if illuminated_scenes_only:
        orig_scene_tokens = get_illuminated_scene_tokens(trucksc)
    else:
        orig_scene_tokens = get_all_scene_tokens(trucksc)

    scene_tokens = []
    for scene_token in orig_scene_tokens:
        scene_record = trucksc.get("scene", scene_token)
        if scene_record["name"] in splits[eval_split]:
            scene_tokens.append(scene_token)
    return scene_tokens


def _next_sample_available(trucksc, sample_data_tokens_list: List[str]) -> bool:
    """
    Check if the next sample is available for all the cameras.

    Args:
        trucksc: TruckScenes database object
        sample_data_tokens_list: List of sample data tokens

    Returns:
        True if next sample is available for all tokens
    """
    for sample_data_token in sample_data_tokens_list:
        sample_data = trucksc.get("sample_data", sample_data_token)
        if sample_data["next"] == "":
            return False
    return True


def get_scene_samples_data(
    trucksc,
    scene_token: str,
    camera_freq: float = 10.0,
) -> List[Dict[str, Any]]:
    """
    Return all sample data dictionaries for a given scene.

    The returned samples are different from the ones in the truckscenes library.
    Here, all the camera frames are collected at the camera frequency, and then
    the radar and lidar measurements are collected as the closest samples to each
    camera frame timestamp.

    Each dictionary in the list contains sensor tokens for all sensors at a given
    timestamp, similar to a sample in the truckscenes library.

    Args:
        trucksc: TruckScenes database object
        scene_token: The token of the scene
        camera_freq: Camera capture frequency in Hz (default 10.0)

    Returns:
        List of dictionaries containing sensor tokens for each timestamp
    """
    # Start by getting the first camera images
    scene_rec = trucksc.get("scene", scene_token)
    first_sample_token = scene_rec["first_sample_token"]
    first_sample = trucksc.get("sample", first_sample_token)

    curr_cameras_sample_data_tokens = [
        token for key, token in first_sample["data"].items() if "CAMERA" in key
    ]

    sensor_types = [
        sensor_type
        for sensor_type in first_sample["data"].keys()
        if "CAMERA" not in sensor_type
    ]

    # Time step based on camera frequency
    time_step = 1 / camera_freq * 1e6  # Convert to microseconds
    curr_time = first_sample["timestamp"]

    sample_data_dict = first_sample["data"].copy()
    samples_data_list = [sample_data_dict]

    while _next_sample_available(trucksc, curr_cameras_sample_data_tokens):
        curr_cameras_sample_data_tokens = [
            trucksc.get("sample_data", token)["next"]
            for token in curr_cameras_sample_data_tokens
        ]
        curr_time += time_step

        # For cameras, only increment the token
        sample_data_dict = {
            key: trucksc.get("sample_data", token)["next"]
            for key, token in sample_data_dict.items()
            if "CAMERA" in key
        }

        # For each radar and lidar, find the closest sample data token to curr_time
        for sensor_type in sensor_types:
            initial_sensor_sample_data_token = first_sample["data"][sensor_type]
            curr_sensor_sample_data_token = initial_sensor_sample_data_token
            curr_sensor_sample_data = trucksc.get(
                "sample_data", curr_sensor_sample_data_token
            )
            while curr_sensor_sample_data["timestamp"] < curr_time:
                curr_sensor_sample_data_token = curr_sensor_sample_data["next"]
                curr_sensor_sample_data = trucksc.get(
                    "sample_data", curr_sensor_sample_data_token
                )
            # curr_sensor_sample_data is in time larger than curr_time.
            # Check which one is closer, this or the previous one
            prev_sensor_sample_data = trucksc.get(
                "sample_data", curr_sensor_sample_data["prev"]
            )
            if abs(curr_time - curr_sensor_sample_data["timestamp"]) < abs(
                curr_time - prev_sensor_sample_data["timestamp"]
            ):
                sample_data_dict[sensor_type] = curr_sensor_sample_data["token"]
            else:
                sample_data_dict[sensor_type] = prev_sensor_sample_data["token"]

        samples_data_list.append(sample_data_dict)

    return samples_data_list


def get_scene_samples(trucksc, scene_token: str) -> List[Dict[str, Any]]:
    """
    Return all keyframe samples for a given scene.

    Each dictionary in the list contains data the same as a sample in the
    truckscenes library.

    Args:
        trucksc: TruckScenes database object
        scene_token: The token of the scene

    Returns:
        List of sample dictionaries for the scene
    """
    scene_rec = trucksc.get("scene", scene_token)
    samples_list = []
    curr_sample_token = scene_rec["first_sample_token"]
    while curr_sample_token != "":
        sample = trucksc.get("sample", curr_sample_token)
        samples_list.append(sample)
        curr_sample_token = sample["next"]
    return samples_list


# Coordinate transformation utilities

def _get_transform_matrix(translation, rotation, inverse: bool = False):
    """
    Create a 4x4 homogeneous transformation matrix.

    Args:
        translation: 3D translation vector
        rotation: Quaternion rotation
        inverse: If True, compute the inverse transform

    Returns:
        4x4 numpy array transformation matrix
    """
    return transform_matrix(translation, Quaternion(rotation), inverse=inverse)


def get_global_from_sensor(trucksc, sensor_sample_data_token: str):
    """
    Get transformation from sensor frame to global frame.

    Args:
        trucksc: TruckScenes database object
        sensor_sample_data_token: Sample data token for the sensor

    Returns:
        Tuple of (translation, rotation_matrix, transformation_matrix)
    """
    sd_rec = trucksc.get("sample_data", sensor_sample_data_token)

    ego_pose_rec = trucksc.get("ego_pose", sd_rec["ego_pose_token"])
    global_from_car = _get_transform_matrix(
        ego_pose_rec["translation"],
        ego_pose_rec["rotation"],
        inverse=False,
    )

    cs_rec = trucksc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    car_from_current = _get_transform_matrix(
        cs_rec["translation"],
        cs_rec["rotation"],
        inverse=False,
    )

    transformation_matrix = reduce(np.dot, [global_from_car, car_from_current])
    translation_matrix = transformation_matrix[:3, 3]
    rotation_matrix = transformation_matrix[:3, :3]

    return translation_matrix, rotation_matrix, transformation_matrix


def get_sensor_from_global(trucksc, sensor_sample_data_token: str):
    """
    Get transformation from global frame to sensor frame.

    Args:
        trucksc: TruckScenes database object
        sensor_sample_data_token: Sample data token for the sensor

    Returns:
        Tuple of (translation, rotation_matrix, transformation_matrix)
    """
    sd_rec = trucksc.get("sample_data", sensor_sample_data_token)

    cs_rec = trucksc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    sd_from_car = _get_transform_matrix(
        cs_rec["translation"],
        cs_rec["rotation"],
        inverse=True,
    )

    ego_pose_rec = trucksc.get("ego_pose", sd_rec["ego_pose_token"])
    car_from_global = _get_transform_matrix(
        ego_pose_rec["translation"],
        ego_pose_rec["rotation"],
        inverse=True,
    )

    transformation_matrix = reduce(np.dot, [sd_from_car, car_from_global])
    translation_matrix = transformation_matrix[:3, 3]
    rotation_matrix = transformation_matrix[:3, :3]

    return translation_matrix, rotation_matrix, transformation_matrix


def get_sensor_from_sensor(
    trucksc,
    source_token: str,
    target_token: str,
):
    """
    Compute transformation from source sensor to target sensor.

    Example: Source sensor would be a camera and target sensor would be
    a reference sensor (e.g., LIDAR_LEFT).

    Args:
        trucksc: TruckScenes database object
        source_token: Sample data token for source sensor
        target_token: Sample data token for target sensor

    Returns:
        Tuple of (translation, rotation_matrix, transformation_matrix)
    """
    _, _, transformation_matrix_g_from_s = get_global_from_sensor(trucksc, source_token)
    _, _, transformation_matrix_t_from_g = get_sensor_from_global(trucksc, target_token)

    transformation_matrix = reduce(
        np.dot, [transformation_matrix_t_from_g, transformation_matrix_g_from_s]
    )
    translation_matrix = transformation_matrix[:3, 3]
    rotation_matrix = transformation_matrix[:3, :3]

    return translation_matrix, rotation_matrix, transformation_matrix


def get_sensor_to_vehicle_flat_up(trucksc, sensor_sample_data_token: str) -> np.ndarray:
    """
    Get transformation from sensor frame to ego-vehicle flat-up frame.

    The flat-up frame is aligned with the ground plane (Z pointing up) and
    oriented with the vehicle's yaw direction.

    Coordinate axes in vehicle flat-up frame:
        - X-axis: Points right
        - Y-axis: Points forward (vehicle heading direction)
        - Z-axis: Points up (perpendicular to ground)

    Args:
        trucksc: TruckScenes database object
        sensor_sample_data_token: Sample data token for the sensor

    Returns:
        4x4 transformation matrix
    """
    sensor_sd_data = trucksc.get("sample_data", sensor_sample_data_token)

    cs_record = trucksc.get(
        "calibrated_sensor", sensor_sd_data["calibrated_sensor_token"]
    )
    pose_record = trucksc.get("ego_pose", sensor_sd_data["ego_pose_token"])

    # Get sensor to ego vehicle transform
    ego_from_ref = _get_transform_matrix(
        translation=cs_record["translation"],
        rotation=cs_record["rotation"],
        inverse=False,
    )

    # Compute rotation between 3D vehicle pose and "flat" vehicle pose
    # (parallel to global z plane).
    ego_yaw = Quaternion(pose_record["rotation"]).yaw_pitch_roll[0]
    rotation_vehicle_flat_from_vehicle = np.dot(
        Quaternion(
            scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]
        ).rotation_matrix,
        Quaternion(pose_record["rotation"]).inverse.rotation_matrix,
    )
    vehicle_flat_from_vehicle = np.eye(4)
    vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
    transformation_matrix = np.dot(vehicle_flat_from_vehicle, ego_from_ref)

    # Rotate upwards
    vehicle_flat_up_from_vehicle_flat = np.eye(4)
    rotation_axis = Quaternion(matrix=transformation_matrix[:3, :3])
    vehicle_flat_up_from_vehicle_flat[:3, :3] = Quaternion(
        axis=rotation_axis.rotate([0, 0, 1]), angle=np.pi / 2
    ).rotation_matrix
    transformation_matrix = np.dot(
        vehicle_flat_up_from_vehicle_flat, transformation_matrix
    )

    return transformation_matrix


def get_radial_velocity(radar_pc: RadarPointCloud) -> np.ndarray:
    """
    Compute radial velocity from radar point cloud velocity components.

    Computes velocity_magnitude * sign(dot(velocity, position)) in the
    sensor's local frame. This is the true Doppler measurement.

    Args:
        radar_pc: RadarPointCloud with points shape (7+, N) where
                  rows 0-2 are positions and 3-5 are velocities

    Returns:
        Radial velocity array of shape (N,)
    """
    pcl = radar_pc.points.copy()
    velocity_magnitude = np.linalg.norm(pcl[3:6, :], axis=0)
    velocity_sign = np.sign(np.sum(pcl[3:6, :] * pcl[:3, :], axis=0))
    return velocity_magnitude * velocity_sign


def load_radar_multisweep(
    trucksc,
    radar_sd_token: str,
    ref_sd_token: str,
    nsweeps: int = 1,
    min_distance: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load radar point cloud with multi-sweep aggregation.

    Loads nsweeps of radar data, transforms positions to the reference frame,
    and computes radial velocity in the sensor frame before transformation.

    Args:
        trucksc: TruckScenes database object
        radar_sd_token: Sample data token for the radar sensor
        ref_sd_token: Sample data token for the reference sensor
        nsweeps: Number of sweeps to aggregate (default=1)
        min_distance: Minimum distance from sensor to keep points (default=0.0)

    Returns:
        Tuple of (points, all_times) where:
        - points: (8, N) array with rows [x, y, z, vx, vy, vz, rcs, radial_velocity]
          Positions (rows 0-2) are in reference frame.
          Velocities (rows 3-5) and radial_velocity (row 7) remain in sensor frame.
        - all_times: (N,) array of time lags in seconds relative to ref_sd_token
    """
    from radargen.radar_maps.transforms import r_pc_transform

    points_list = []
    times_list = []

    # Get reference pose and timestamp.
    ref_sd_rec = trucksc.get('sample_data', ref_sd_token)
    ref_pose_rec = trucksc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = trucksc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    ref_time = ref_sd_rec['timestamp']

    # Homogeneous transform from ego car frame to reference frame.
    ref_from_car = transform_matrix(ref_cs_rec['translation'],
                                    Quaternion(ref_cs_rec['rotation']),
                                    inverse=True)

    # Homogeneous transformation matrix from global to _current_ ego car frame.
    car_from_global = transform_matrix(ref_pose_rec['translation'],
                                        Quaternion(ref_pose_rec['rotation']),
                                        inverse=True)

    current_sd_token = radar_sd_token

    for _ in range(nsweeps):
        if current_sd_token == "":
            break

        current_sd_rec = trucksc.get("sample_data", current_sd_token)

        # Load radar point cloud
        radar_file = osp.join(trucksc.dataroot, current_sd_rec["filename"])
        current_pc = RadarPointCloud.from_file(radar_file)

        if current_pc.nbr_points() == 0:
            raise RuntimeError(f"No points found in radar file {radar_file}, this is unexpected.")
            # To skip use:
            # current_sd_token = current_sd["prev"]
            # continue

        # Remove points too close to sensor
        distances = np.linalg.norm(current_pc.points[:3, :], axis=0)
        keep_mask = distances >= min_distance
        current_pc.points = current_pc.points[:, keep_mask]

        if current_pc.nbr_points() == 0:
            raise RuntimeError(f"No points left after filtering in radar file {radar_file}, this is unexpected.")
            # To skip use:
            # current_sd_token = current_sd["prev"]
            # continue

        # Compute radial velocity in sensor frame BEFORE position transform
        radial_vel = get_radial_velocity(current_pc)

        # Get past pose.
        current_pose_rec = trucksc.get('ego_pose', current_sd_rec['ego_pose_token'])
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                            Quaternion(current_pose_rec['rotation']),
                                            inverse=False)

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        current_cs_rec = trucksc.get('calibrated_sensor',
                                        current_sd_rec['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs_rec['translation'],
                                            Quaternion(current_cs_rec['rotation']),
                                            inverse=False)

        # Fuse four transformation matrices into one and perform transform.
        trans_matrix = reduce(np.dot, [ref_from_car, car_from_global,
                                        global_from_car, car_from_current])
        # current_pc.transform(trans_matrix)
        current_pc = r_pc_transform(current_pc, trans_matrix)

        # Compute time lag
        time_lag = (current_sd_rec["timestamp"] - ref_time) * 1e-6  # seconds

        # Build 8-row array: [x, y, z, vx, vy, vz, rcs, radial_velocity]
        n_pts = current_pc.nbr_points()
        sweep_points = np.zeros((8, n_pts), dtype=np.float64)
        sweep_points[:3, :] = current_pc.points[:3, :]  # positions (ref frame)
        sweep_points[3:6, :] = current_pc.points[3:6, :]  # velocities (sensor frame)
        sweep_points[6, :] = current_pc.points[6, :]  # rcs
        sweep_points[7, :] = radial_vel  # radial velocity (sensor frame)

        points_list.append(sweep_points)
        times_list.append(np.full(n_pts, time_lag))

        # Move to previous sweep
        current_sd_token = current_sd_rec["prev"]

    if len(points_list) == 0:
        return np.zeros((8, 0), dtype=np.float64), np.zeros(0, dtype=np.float64)

    all_points = np.hstack(points_list)
    all_times = np.concatenate(times_list)

    return all_points, all_times

