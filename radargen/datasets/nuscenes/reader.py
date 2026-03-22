"""
nuScenes data reading utilities.
"""

import os.path as osp
from functools import reduce
from typing import Any, Dict, List, Tuple

import numpy as np
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion

from radargen.datasets.nuscenes.config import NUSCENES_CAMERA_VIEWS


def get_split_scene_tokens(nusc, eval_split: str) -> List[str]:
    """
    Returns the scene tokens for the given split.

    Args:
        nusc: NuScenes database object
        eval_split: Split name (e.g., "train", "val", "mini_train", "mini_val")

    Returns:
        List of scene token strings for the split
    """
    splits = create_splits_scenes()
    if eval_split not in splits:
        raise ValueError(
            f"Unknown split '{eval_split}'. Available splits: {list(splits.keys())}"
        )
    split_scene_names = set(splits[eval_split])

    scene_tokens = []
    for scene in nusc.scene:
        if scene["name"] in split_scene_names:
            scene_tokens.append(scene["token"])
    return scene_tokens


def get_scene_keyframe_samples(nusc, scene_token: str) -> List[Dict[str, Any]]:
    """
    Return all keyframe samples for a given scene by following the sample linked list.

    Each keyframe corresponds to an annotated sample (~2 Hz). The returned dicts
    contain sensor channel -> sample_data_token mappings plus metadata keys.

    Args:
        nusc: NuScenes database object
        scene_token: The token of the scene

    Returns:
        List of dicts with sensor tokens and metadata:
        {
            "CAM_FRONT": <sample_data_token>,
            ...,
            "LIDAR_TOP": <sample_data_token>,
            "RADAR_FRONT": <sample_data_token>,
            ...,
            "_sample_token": <sample_token>,
            "_timestamp": <timestamp_us>,
        }
    """
    scene = nusc.get("scene", scene_token)
    samples_list = []
    curr_sample_token = scene["first_sample_token"]
    while curr_sample_token != "":
        sample = nusc.get("sample", curr_sample_token)
        sample_dict = dict(sample["data"])  # channel -> sample_data_token
        sample_dict["_sample_token"] = sample["token"]
        sample_dict["_timestamp"] = sample["timestamp"]
        samples_list.append(sample_dict)
        curr_sample_token = sample["next"]
    return samples_list


def _advance_cameras(nusc, curr_cam_tokens: Dict[str, str]):
    """
    Advance all cameras one step along their sample_data chains.

    Returns the next token dict if every camera has a next frame, or None if
    any camera has reached the end of its chain.
    """
    next_tokens = {}
    for cam_name, cam_token in curr_cam_tokens.items():
        cam_sd = nusc.get("sample_data", cam_token)
        if cam_sd["next"] == "":
            return None
        next_tokens[cam_name] = cam_sd["next"]
    return next_tokens


def get_scene_all_samples(
    nusc,
    scene_token: str,
    camera_freq: float = 12.0,
) -> List[Dict[str, Any]]:
    """
    Return all sample data dicts for a given scene at camera frequency (~12 Hz).

    Follows camera sample_data['next'] pointers for all frames, and for each
    camera frame finds the temporally closest radar/lidar sample_data using the
    actual camera timestamp (not a synthetic time estimate).

    Non-camera sensor cursors advance monotonically — they never restart from the
    beginning, making the function O(M + N) instead of O(M * N).

    Args:
        nusc: NuScenes database object
        scene_token: The token of the scene
        camera_freq: Unused; kept for API compatibility.

    Returns:
        List of dicts containing sensor tokens for each camera timestamp
    """
    scene_rec = nusc.get("scene", scene_token)
    first_sample = nusc.get("sample", scene_rec["first_sample_token"])

    non_camera_types = [s for s in first_sample["data"] if "CAM" not in s]
    curr_cam_tokens = {k: v for k, v in first_sample["data"].items() if "CAM" in k}
    sensor_current_token = {s: first_sample["data"][s] for s in non_camera_types}

    samples_data_list = [dict(first_sample["data"])]

    curr_cam_tokens = _advance_cameras(nusc, curr_cam_tokens)
    while curr_cam_tokens is not None:
        # Use the actual CAM_FRONT hardware timestamp as reference
        target_time = nusc.get("sample_data", curr_cam_tokens["CAM_FRONT"])["timestamp"]

        sample_data_dict = dict(curr_cam_tokens)

        for sensor in non_camera_types:
            curr_token = sensor_current_token[sensor]
            curr_sd = nusc.get("sample_data", curr_token)
            # Walk the persistent cursor forward until at or past target_time
            while curr_sd["next"] != "" and curr_sd["timestamp"] < target_time:
                curr_token = curr_sd["next"]
                curr_sd = nusc.get("sample_data", curr_token)
            # Pick closest: compare current position with previous
            if curr_sd["prev"] != "":
                prev_sd = nusc.get("sample_data", curr_sd["prev"])
                if abs(target_time - prev_sd["timestamp"]) < abs(
                    target_time - curr_sd["timestamp"]
                ):
                    best_token = prev_sd["token"]
                    sensor_current_token[sensor] = prev_sd["token"]
                else:
                    best_token = curr_token
                    sensor_current_token[sensor] = curr_token
            else:
                best_token = curr_token
                sensor_current_token[sensor] = curr_token
            sample_data_dict[sensor] = best_token

        samples_data_list.append(sample_data_dict)
        curr_cam_tokens = _advance_cameras(nusc, curr_cam_tokens)

    return samples_data_list


def _get_transform_matrix(translation, rotation, inverse: bool = False) -> np.ndarray:
    """Create a 4x4 homogeneous transformation matrix."""
    return transform_matrix(translation, Quaternion(rotation), inverse=inverse)


def get_sensor_to_reference_transform(
    nusc,
    source_sd_token: str,
    target_sd_token: str,
) -> np.ndarray:
    """
    Compute 4x4 transformation from source sensor frame to target sensor frame.

    Chains: ref_from_car @ car_from_global @ global_from_car @ car_from_source

    Args:
        nusc: NuScenes database object
        source_sd_token: Sample data token for the source sensor
        target_sd_token: Sample data token for the target (reference) sensor

    Returns:
        4x4 transformation matrix (source -> target)
    """
    # Source: sensor -> global
    src_sd = nusc.get("sample_data", source_sd_token)
    src_pose = nusc.get("ego_pose", src_sd["ego_pose_token"])
    src_cs = nusc.get("calibrated_sensor", src_sd["calibrated_sensor_token"])

    global_from_car_src = _get_transform_matrix(
        src_pose["translation"], src_pose["rotation"], inverse=False
    )
    car_from_sensor_src = _get_transform_matrix(
        src_cs["translation"], src_cs["rotation"], inverse=False
    )
    global_from_source = reduce(np.dot, [global_from_car_src, car_from_sensor_src])

    # Target: global -> sensor
    tgt_sd = nusc.get("sample_data", target_sd_token)
    tgt_pose = nusc.get("ego_pose", tgt_sd["ego_pose_token"])
    tgt_cs = nusc.get("calibrated_sensor", tgt_sd["calibrated_sensor_token"])

    car_from_global_tgt = _get_transform_matrix(
        tgt_pose["translation"], tgt_pose["rotation"], inverse=True
    )
    sensor_from_car_tgt = _get_transform_matrix(
        tgt_cs["translation"], tgt_cs["rotation"], inverse=True
    )
    target_from_global = reduce(np.dot, [sensor_from_car_tgt, car_from_global_tgt])

    return reduce(np.dot, [target_from_global, global_from_source])


def get_sensor_to_vehicle_flat_up(nusc, ref_sd_token: str) -> np.ndarray:
    """
    Get 4x4 transformation from reference sensor frame to vehicle flat-up frame.

    The flat-up frame is aligned with the ground plane (Z pointing up) and
    oriented with the vehicle's yaw direction.

    Coordinate axes in vehicle flat-up frame:
        - X-axis: Points right
        - Y-axis: Points forward (vehicle heading direction)
        - Z-axis: Points up (perpendicular to ground)

    Args:
        nusc: NuScenes database object
        ref_sd_token: Sample data token for the reference sensor

    Returns:
        4x4 transformation matrix
    """
    sensor_sd_data = nusc.get("sample_data", ref_sd_token)

    cs_record = nusc.get("calibrated_sensor", sensor_sd_data["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sensor_sd_data["ego_pose_token"])

    # Sensor to ego vehicle transform
    ego_from_ref = _get_transform_matrix(
        translation=cs_record["translation"],
        rotation=cs_record["rotation"],
        inverse=False,
    )

    # Compute rotation between 3D vehicle pose and "flat" vehicle pose
    # (parallel to global z plane)
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

    # Rotate upwards (90 degrees around the forward axis)
    vehicle_flat_up_from_vehicle_flat = np.eye(4)
    rotation_axis = Quaternion(matrix=transformation_matrix[:3, :3])
    vehicle_flat_up_from_vehicle_flat[:3, :3] = Quaternion(
        axis=rotation_axis.rotate([0, 0, 1]), angle=np.pi / 2
    ).rotation_matrix
    transformation_matrix = np.dot(
        vehicle_flat_up_from_vehicle_flat, transformation_matrix
    )

    return transformation_matrix


def load_radar_multisweep(
    nusc,
    radar_sd_token: str,
    ref_sd_token: str,
    nsweeps: int = 1,
    min_distance: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load nuScenes radar point cloud with multi-sweep aggregation.

    Loads nsweeps of radar data (18-column nuScenes format), extracts relevant
    fields, computes radial velocity in sensor frame before transformation, then
    transforms positions to the reference sensor frame.

    nuScenes radar PCL column layout (18 cols):
        0: x, 1: y, 2: z,
        3: dyn_prop, 4: id,
        5: rcs,
        6: vx (raw, ego-relative), 7: vy (raw, ego-relative),
        8: vx_comp, 9: vy_comp,
        10: is_quality_valid, 11: ambig_state,
        12: x_rms, 13: y_rms, 14: invalid_state, 15: pdh0,
        16: vx_rms, 17: vy_rms

    Args:
        nusc: NuScenes database object
        radar_sd_token: Sample data token for the radar sensor
        ref_sd_token: Sample data token for the reference sensor (LIDAR_TOP)
        nsweeps: Number of sweeps to aggregate (default=1)
        min_distance: Minimum distance from sensor to keep points (default=0.0)

    Returns:
        Tuple of (points, all_times) where:
        - points: (8, N) array with rows [x, y, z, vx, vy, vz, rcs, radial_velocity]
          Positions (rows 0-2) are in reference frame.
          Velocities (rows 3-5) and radial_velocity (row 7) remain in sensor frame.
          vz is always 0 (nuScenes radar is 2D).
        - all_times: (N,) array of time lags in seconds relative to ref_sd_token
    """
    points_list = []
    times_list = []

    # Get reference pose and timestamp
    ref_sd_rec = nusc.get("sample_data", ref_sd_token)
    ref_pose_rec = nusc.get("ego_pose", ref_sd_rec["ego_pose_token"])
    ref_cs_rec = nusc.get("calibrated_sensor", ref_sd_rec["calibrated_sensor_token"])
    ref_time = ref_sd_rec["timestamp"]

    # Homogeneous transform from ego car frame to reference frame
    ref_from_car = _get_transform_matrix(
        ref_cs_rec["translation"], ref_cs_rec["rotation"], inverse=True
    )

    # Homogeneous transform from global to current ego car frame
    car_from_global = _get_transform_matrix(
        ref_pose_rec["translation"], ref_pose_rec["rotation"], inverse=True
    )

    current_sd_token = radar_sd_token

    for _ in range(nsweeps):
        if current_sd_token == "":
            break

        current_sd_rec = nusc.get("sample_data", current_sd_token)

        # Load nuScenes radar point cloud (18 columns, filtered by default states)
        radar_file = osp.join(nusc.dataroot, current_sd_rec["filename"])
        current_pc = RadarPointCloud.from_file(radar_file)

        if current_pc.nbr_points() == 0:
            current_sd_token = current_sd_rec["prev"]
            continue

        # Remove points too close to sensor
        if min_distance > 0.0:
            distances = np.linalg.norm(current_pc.points[:2, :], axis=0)
            keep_mask = distances >= min_distance
            current_pc.points = current_pc.points[:, keep_mask]

        if current_pc.nbr_points() == 0:
            current_sd_token = current_sd_rec["prev"]
            continue

        # Extract raw velocities (sensor frame, ego-relative Doppler measurement)
        # Use vx (index 6) and vy (index 7) — NOT compensated velocities
        vx = current_pc.points[6, :]  # raw vx
        vy = current_pc.points[7, :]  # raw vy

        # Compute radial velocity in sensor frame BEFORE position transform
        # radial_vel = ||[vx, vy]|| * sign(vx*x + vy*y)
        x_pos = current_pc.points[0, :]
        y_pos = current_pc.points[1, :]
        velocity_magnitude = np.sqrt(vx ** 2 + vy ** 2)
        velocity_sign = np.sign(vx * x_pos + vy * y_pos)
        radial_vel = velocity_magnitude * velocity_sign

        # Get past pose for this sweep
        current_pose_rec = nusc.get("ego_pose", current_sd_rec["ego_pose_token"])
        global_from_car = _get_transform_matrix(
            current_pose_rec["translation"], current_pose_rec["rotation"], inverse=False
        )

        # Sensor to ego car frame
        current_cs_rec = nusc.get(
            "calibrated_sensor", current_sd_rec["calibrated_sensor_token"]
        )
        car_from_current = _get_transform_matrix(
            current_cs_rec["translation"], current_cs_rec["rotation"], inverse=False
        )

        # Fuse transforms: sensor -> ego -> global -> ref_ego -> ref_sensor
        trans_matrix = reduce(
            np.dot,
            [ref_from_car, car_from_global, global_from_car, car_from_current],
        )

        # Transform positions (x, y, z) to reference frame
        n_pts = current_pc.nbr_points()
        pos_hom = np.vstack(
            [current_pc.points[:3, :], np.ones((1, n_pts))]
        )
        pos_ref = (trans_matrix @ pos_hom)[:3, :]

        # Compute time lag in seconds
        time_lag = (current_sd_rec["timestamp"] - ref_time) * 1e-6

        # Build 8-row array: [x, y, z, vx, vy, vz, rcs, radial_velocity]
        sweep_points = np.zeros((8, n_pts), dtype=np.float64)
        sweep_points[0:3, :] = pos_ref          # positions in reference frame
        sweep_points[3, :] = vx                  # raw vx (sensor frame)
        sweep_points[4, :] = vy                  # raw vy (sensor frame)
        sweep_points[5, :] = 0.0                 # vz = 0 (2D radar)
        sweep_points[6, :] = current_pc.points[5, :]  # rcs (index 5)
        sweep_points[7, :] = radial_vel          # radial velocity (sensor frame)

        points_list.append(sweep_points)
        times_list.append(np.full(n_pts, time_lag))

        # Move to previous sweep
        current_sd_token = current_sd_rec["prev"]

    if len(points_list) == 0:
        return np.zeros((8, 0), dtype=np.float64), np.zeros(0, dtype=np.float64)

    all_points = np.hstack(points_list)
    all_times = np.concatenate(times_list)

    return all_points, all_times
