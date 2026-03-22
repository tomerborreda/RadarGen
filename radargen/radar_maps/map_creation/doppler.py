"""Doppler velocity map creation and conversion functions."""

import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors

#### Conversion functions ####

def doppler_map_to_image(doppler_map: np.ndarray, doppler_min: float, doppler_max: float) -> tuple[Image.Image, np.ndarray]:
    """Convert a Doppler velocity map to image representation.

    Returns both a PIL Image for visualization and a normalized float array
    for further processing.

    Args:
        doppler_map: Doppler map as (H, W) array with values in [doppler_min, doppler_max]
        doppler_min: Minimum Doppler velocity in m/s
        doppler_max: Maximum Doppler velocity in m/s

    Returns:
        Tuple of:
        - PIL Image (uint8 RGB)
        - Normalized float array (H, W, 3) in range [0, 255]

    Raises:
        ValueError: If input array is empty
    """
    if doppler_map.size == 0:
        raise ValueError("Doppler map is empty")

    # Normalize to 0-1 range using Doppler bounds
    image = (doppler_map - doppler_min) / (doppler_max - doppler_min)
    assert np.all(image >= 0.0) and np.all(image <= 1.0), f"Doppler values are out of bounds: [{doppler_map.min()}, {doppler_map.max()}]. Image values are not in the range [0, 1]"
    # Scale to 0-255
    image = image * 255.0

    # Convert grayscale to RGB (3 channels)
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
        image = np.repeat(image, 3, axis=-1)

    # Return both PIL Image (uint8) and float array for downstream processing
    return Image.fromarray(image.astype(np.uint8)), image


def decoded_to_doppler_map(decoded_image: np.ndarray, doppler_min: float, doppler_max: float) -> np.ndarray:
    """Convert a decoded RGB image back to a Doppler velocity map.

    This is approximately the inverse of doppler_map_to_image().

    Args:
        decoded_image: RGB image as (H, W, 3) array
        doppler_min: Minimum Doppler velocity in m/s
        doppler_max: Maximum Doppler velocity in m/s

    Returns:
        Doppler map as (H, W) array with values in [doppler_min, doppler_max]

    Raises:
        ValueError: If image is empty
    """
    if decoded_image.size == 0:
        raise ValueError("Decoded image is empty")

    # Average over RGB channels to get grayscale
    image = np.mean(decoded_image, axis=-1)

    # Denormalize from 0-255 to Doppler range
    doppler_map = ((image / 255.0) * (doppler_max - doppler_min)) + doppler_min

    return doppler_map


#### Map creation functions ####

def create_raw_doppler_map(
        points: np.ndarray,
        radial_velocity: np.ndarray,
        image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a sparse Doppler velocity map from point locations.

    For pixels with multiple points, keeps the value with maximum absolute velocity.
    Uses bottom-left origin coordinate system (y increases upward).

    Args:
        points: Point locations as (2, N) array where rows are [x, y]
        radial_velocity: Radial velocity values as (N,) array
        image_size: Size of output square map (both height and width)

    Returns:
        Tuple of:
        - doppler_map: Sparse Doppler map as (height, width) array with velocities
        - doppler_map_mask: Boolean mask as (height, width) where True indicates
                           a point exists at that pixel

    Raises:
        ValueError: If input arrays have incorrect shapes or mismatched sizes
    """
    # Validate inputs
    if points.ndim != 2 or points.shape[0] != 2:
        raise ValueError(f"Points must have shape (2, N), got {points.shape}")
    if radial_velocity.ndim != 1:
        raise ValueError(f"Radial velocity must be 1D, got shape {radial_velocity.shape}")
    if points.shape[1] != radial_velocity.shape[0]:
        raise ValueError(
            f"Number of points ({points.shape[1]}) doesn't match "
            f"number of velocities ({radial_velocity.shape[0]})"
        )

    height = width = image_size

    # Initialize output arrays
    doppler_map = np.zeros((height, width), dtype=np.float64)
    doppler_map_mask = np.zeros((height, width), dtype=bool)

    if len(radial_velocity) > 0:
        # Convert to pixel indices
        x_int = np.floor(points[0]).astype(np.int32)
        y_int = height - 1 - np.floor(points[1]).astype(np.int32)

        # Filter valid points within bounds
        valid = (x_int >= 0) & (x_int < width) & (y_int >= 0) & (y_int < height)
        x_valid = x_int[valid]
        y_valid = y_int[valid]
        vel_valid = radial_velocity[valid]
        abs_vel_valid = np.abs(vel_valid)

        # Convert to linear indices
        linear_idx = np.ravel_multi_index((y_valid, x_valid), (height, width))

        # Sort by pixel index, then by descending absolute velocity
        # This groups points by pixel and puts highest abs velocity first
        sort_order = np.lexsort((-abs_vel_valid, linear_idx))
        linear_idx_sorted = linear_idx[sort_order]
        vel_sorted = vel_valid[sort_order]

        # Keep first occurrence (highest abs velocity) for each unique pixel
        unique_pixels, first_idx = np.unique(linear_idx_sorted, return_index=True)

        doppler_map.ravel()[unique_pixels] = vel_sorted[first_idx]
        doppler_map_mask.ravel()[unique_pixels] = True

    return doppler_map, doppler_map_mask


def create_interpolated_doppler_map(
        doppler_map: np.ndarray,
        doppler_map_mask: np.ndarray,
        image_size: int,
        max_interpolation_distance: float = None,
) -> np.ndarray:
    """Fill sparse Doppler map using nearest neighbor interpolation.

    Pixels far from any known point are set to zero (no velocity).

    Args:
        doppler_map: Sparse Doppler map as (height, width) array
        doppler_map_mask: Boolean mask as (height, width) indicating known points
        image_size: Size of the map (height and width)
        max_interpolation_distance: Maximum distance in pixels for interpolation.
                                   Points farther than this are set to 0.
                                   Default: image_size / 10.0

    Returns:
        Interpolated Doppler map as (height, width) array

    Raises:
        ValueError: If no known points exist for interpolation
    """
    # Validate inputs
    if doppler_map.shape != (image_size, image_size):
        raise ValueError(
            f"Doppler map shape {doppler_map.shape} doesn't match "
            f"image_size ({image_size}, {image_size})"
        )
    if doppler_map_mask.shape != doppler_map.shape:
        raise ValueError("Doppler map and mask must have same shape")

    # Check if we have any known points
    if not np.any(doppler_map_mask):
        raise ValueError(
            "Cannot interpolate Doppler map: no known points. "
            "This should have been caught upstream in generator.py."
        )

    # Set default max distance
    if max_interpolation_distance is None:
        max_interpolation_distance = image_size / 10.0

    height = width = image_size

    # Create coordinate grids for all pixels
    grid_x, grid_y = np.meshgrid(
        np.arange(width),
        np.arange(height),
        indexing='xy'
    )

    # Extract known point coordinates and values
    known_y, known_x = np.where(doppler_map_mask)
    known_points = np.column_stack((known_x, known_y))
    known_values = doppler_map[doppler_map_mask]

    # Query points: all pixel locations
    query_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    # Find nearest neighbor for each pixel
    nbrs = NearestNeighbors(n_neighbors=1).fit(known_points)
    distances, indices = nbrs.kneighbors(query_points)

    # Assign velocity from nearest neighbor
    interp_doppler_flat = known_values[indices[:, 0]]

    # Zero out pixels that are too far from any known point
    is_too_far = distances[:, 0] > max_interpolation_distance
    interp_doppler_flat[is_too_far] = 0.0

    # Validate no NaN values
    if np.any(np.isnan(interp_doppler_flat)):
        raise RuntimeError("Interpolation produced NaN values")

    # Reshape to image dimensions
    interp_doppler = interp_doppler_flat.reshape(height, width)

    return interp_doppler


def create_radar_doppler_map(
        locations: np.ndarray,
        radial_velocity: np.ndarray,
        image_size: int,
        max_interpolation_distance: float = None,
) -> np.ndarray:
    """Create an interpolated Doppler velocity map from 2D point locations.

    Combines sparse Doppler map creation with nearest neighbor interpolation to
    produce a dense velocity map.

    Args:
        locations: Point locations as (2, N) array where rows are [x, y]
        radial_velocity: Radial velocity values as (N,) array
        image_size: Size of output square map (both height and width)
        max_interpolation_distance: Maximum distance in pixels for interpolation.
                                   Default: image_size / 10.0

    Returns:
        Interpolated Doppler map as (image_size, image_size) array

    Raises:
        ValueError: If locations array has incorrect shape or is empty
    """
    # Validate input shape
    if locations.ndim != 2 or locations.shape[0] != 2:
        raise ValueError(f"Locations must have shape (2, N), got {locations.shape}")

    # Create sparse Doppler map
    raw_doppler_map, doppler_map_mask = create_raw_doppler_map(
        locations, radial_velocity, image_size
    )

    # Interpolate to fill gaps
    return create_interpolated_doppler_map(
        raw_doppler_map,
        doppler_map_mask,
        image_size,
        max_interpolation_distance=max_interpolation_distance,
    )
