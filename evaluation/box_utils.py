"""Box geometry utilities for point-in-box tests and coordinate transforms."""

import numpy as np
from matplotlib.path import Path


def get_points_in_box(pcl: np.ndarray, bottom_corners: np.ndarray):
    """Filter a point cloud to keep only points inside a 2D box polygon.

    Works for arbitrarily rotated boxes.

    Args:
        pcl: (N, D) array. Columns 0 and 1 are x and y.
        bottom_corners: (3, 4) array of the 4 bottom corners of the box.

    Returns:
        filtered_pcl: (M, D) array of points inside the box.
        mask: (N,) boolean mask.
    """
    corners_xy = bottom_corners[0:2, :].T  # (4, 2)
    box_path = Path(corners_xy)
    pcl_xy = pcl[:, 0:2]
    mask = box_path.contains_points(pcl_xy)
    return pcl[mask], mask


def convert_pcl_to_box_coords(pcl, box):
    """Transform a 2D point cloud into the box's local coordinate system.

    The origin is the box center; axes are aligned with the box orientation.

    Args:
        pcl: (N, D) array. Columns 0-1 are x, y; remaining columns are kept.
        box: Box object with ``.center`` (3,), ``.rotation_matrix`` (3, 3),
             and ``.wlh`` (3,).

    Returns:
        pcl_box_coords: (N, D) array in box-local coordinates.
        width: Box width (local y-axis extent).
        length: Box length (local x-axis extent).
    """
    extra_coords = None
    if pcl.shape[1] > 2:
        extra_coords = pcl[:, 2:]
        pcl = pcl[:, :2]

    pcl_centered = pcl - box.center[:2]
    R_box_to_world_2D = box.rotation_matrix[:2, :2]
    pcl_box_coords = pcl_centered @ R_box_to_world_2D

    width = box.wlh[0]
    length = box.wlh[1]

    if extra_coords is not None:
        pcl_box_coords = np.hstack([pcl_box_coords, extra_coords])

    return pcl_box_coords, width, length
