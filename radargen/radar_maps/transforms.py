"""Coordinate transformation utilities for radar map generation."""

from functools import reduce

import numpy as np
from pyquaternion import Quaternion
from truckscenes import TruckScenes
from truckscenes.utils.geometry_utils import transform_matrix
from truckscenes.utils.data_classes import RadarPointCloud


def r_pc_rotate(r_pc: RadarPointCloud, rot_matrix: np.ndarray) -> RadarPointCloud:
    """
    Applies a rotation.
    :param rot_matrix: <np.float: 3, 3>. Rotation matrix.
    """
    # Rotate the point cloud positions
    r_pc.points[:3, :] = np.dot(rot_matrix, r_pc.points[:3, :])
    # Rotate the velocity vectors
    r_pc.points[3:6, :] = np.dot(rot_matrix, r_pc.points[3:6, :])
    return r_pc


def pc_transform(pcl: np.ndarray, transf_matrix: np.ndarray) -> np.ndarray:
    """
    Applies a homogeneous transform.
    pcl: <np.float: N, 3>. Point cloud.
    :param transf_matrix: <np.float: 4, 4>. Homogenous transformation matrix.
    """
    pcl = pcl.T
    # Transform the point cloud positions
    pcl = transf_matrix.dot(
        np.vstack((pcl, np.ones(pcl.shape[1])))
    )[:3, :]
    return pcl.T


def r_pc_transform(r_pc: RadarPointCloud, transf_matrix: np.ndarray) -> RadarPointCloud:
    """
    Applies a homogeneous transform.
    :param transf_matrix: <np.float: 4, 4>. Homogenous transformation matrix.
    """
    # Transform the point cloud positions
    r_pc.points[:3, :] = transf_matrix.dot(
        np.vstack((r_pc.points[:3, :], np.ones(r_pc.nbr_points())))
    )[:3, :]

    # NOTE: We use the radial velocity of each point in the sensor's frame.
    # The velocity components are of the radial velocity in the sensor's frame and not the full velocity.
    # Converting to a different frame would make the doppler value for the points incorrect if calculated from the new origin.

    # Transform the velocity vectors
    # r_pc.points[3:6, :] = np.dot(np.hstack((transf_matrix[:, :3], np.identity(4)[:, 3][:, None])),
    #     np.vstack((r_pc.points[3:6, :], np.ones(r_pc.nbr_points())))
    # )[:3, :]
    return r_pc


def global_from_sensor(trucksc: TruckScenes, sensor_sample_data_token: str):
    sd_rec = trucksc.get('sample_data', sensor_sample_data_token)

    ego_pose_rec = trucksc.get('ego_pose', sd_rec['ego_pose_token'])
    global_from_car = transform_matrix(ego_pose_rec['translation'],
                                       Quaternion(ego_pose_rec['rotation']),
                                       inverse=False)

    # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
    cs_rec = trucksc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    car_from_current = transform_matrix(cs_rec['translation'],
                                        Quaternion(cs_rec['rotation']),
                                        inverse=False)

    transformation_matrix = reduce(np.dot, [global_from_car, car_from_current])
    translation_matrix = transformation_matrix[:3, 3]
    rotation_matrix = transformation_matrix[:3, :3]

    return translation_matrix, rotation_matrix, transformation_matrix


def sensor_from_global(trucksc: TruckScenes, sensor_sample_data_token: str):
    sd_rec = trucksc.get('sample_data', sensor_sample_data_token)

    # Homogeneous transformation matrix from ego car frame to sensor coordinate frame.
    cs_rec = trucksc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    sd_from_car = transform_matrix(cs_rec['translation'],
                                    Quaternion(cs_rec['rotation']),
                                    inverse=True)

    ego_pose_rec = trucksc.get('ego_pose', sd_rec['ego_pose_token'])
    car_from_global = transform_matrix(ego_pose_rec['translation'],
                                       Quaternion(ego_pose_rec['rotation']),
                                       inverse=True)

    transformation_matrix = reduce(np.dot, [sd_from_car, car_from_global])
    translation_matrix = transformation_matrix[:3, 3]
    rotation_matrix = transformation_matrix[:3, :3]

    return translation_matrix, rotation_matrix, transformation_matrix


def sensor_from_sensor(trucksc: TruckScenes, sensor_sample_data_token_s: str, sensor_sample_data_token_t: str):
    """
    Compute the translation and rotation matrices for conversion from source sensor to target sensor.

    Example: Source sensor would be a camera and target sensor would be a reference sensor (e.g. LIDAR_LEFT) at the first frame.

    Args:
        trucksc: TruckScenes object
        sensor_sample_data_token_s: str, token of source sensor
        sensor_sample_data_token_t: str, token of target sensor

    Returns:
        translation_matrix: np.ndarray, 3x1 translation matrix
        rotation_matrix: np.ndarray, 3x3 rotation matrix
    """
    # Get the transformation matrices from the source and target sensors to the global frame
    _, _, transformation_matrix_g_from_s = global_from_sensor(trucksc, sensor_sample_data_token_s)
    _, _, transformation_matrix_t_from_g = sensor_from_global(trucksc, sensor_sample_data_token_t)

    # Compute the transformation matrix from the source sensor to the target sensor
    transformation_matrix = reduce(np.dot, [transformation_matrix_t_from_g, transformation_matrix_g_from_s])
    translation_matrix = transformation_matrix[:3, 3]
    rotation_matrix = transformation_matrix[:3, :3]

    return translation_matrix, rotation_matrix, transformation_matrix


def sensor_to_vehicle_flat_up(trucksc: TruckScenes, sensor_sample_data_token: str):
    sensor_sd_data = trucksc.get('sample_data', sensor_sample_data_token)

    # Retrieve transformation matrices for sensor point cloud.
    cs_record = trucksc.get('calibrated_sensor',
                            sensor_sd_data['calibrated_sensor_token'])
    pose_record = trucksc.get('ego_pose',
                                sensor_sd_data['ego_pose_token'])
    ego_from_ref = transform_matrix(translation=cs_record['translation'],
                                    rotation=Quaternion(cs_record["rotation"]),
                                    inverse=False)

    # Compute rotation between 3D vehicle pose and "flat" vehicle pose
    # (parallel to global z plane).
    ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
    rotation_vehicle_flat_from_vehicle = np.dot(
        Quaternion(scalar=np.cos(ego_yaw / 2),
                    vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
        Quaternion(pose_record['rotation']).inverse.rotation_matrix)
    vehicle_flat_from_vehicle = np.eye(4)
    vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
    transformation_matrix = np.dot(vehicle_flat_from_vehicle, ego_from_ref)

    # Rotate upwards
    vehicle_flat_up_from_vehicle_flat = np.eye(4)
    rotation_axis = Quaternion(matrix=transformation_matrix[:3, :3])
    vehicle_flat_up_from_vehicle_flat[:3, :3] = \
        Quaternion(axis=rotation_axis.rotate([0, 0, 1]),
                    angle=np.pi/2).rotation_matrix
    transformation_matrix = np.dot(vehicle_flat_up_from_vehicle_flat, transformation_matrix)

    return transformation_matrix


def transform_r_pcl_from_ref_to_vehicle_flat_up(
        trucksc: TruckScenes,
        pc: RadarPointCloud,
        ref_sd_token: str
) -> RadarPointCloud:
    transformation_matrix = sensor_to_vehicle_flat_up(trucksc, ref_sd_token)
    pc = r_pc_transform(pc, transformation_matrix)
    return pc


def transform_pcl_from_ref_to_vehicle_flat_up(
        trucksc: TruckScenes,
        pc: np.ndarray,
        ref_sd_token: str
) -> np.ndarray:
    transformation_matrix = sensor_to_vehicle_flat_up(trucksc, ref_sd_token)
    pc = pc_transform(pc, transformation_matrix)
    return pc


def _invert_rigid_transformation(T: np.ndarray) -> np.ndarray:
    """
    Fast inverse of a 4x4 rigid transform (SE(3)).
    """
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    Rt = R.T
    T_inv[:3, :3] = Rt
    T_inv[:3, 3] = -Rt @ t
    return T_inv


def transform_r_pcl_from_vehicle_flat_up_to_ref(
        trucksc: TruckScenes,
        pc: RadarPointCloud,
        ref_sd_token: str
) -> RadarPointCloud:
    transformation_matrix = sensor_to_vehicle_flat_up(trucksc, ref_sd_token)
    inverse_transformation_matrix = _invert_rigid_transformation(transformation_matrix)
    pc = r_pc_transform(pc, inverse_transformation_matrix)
    return pc


