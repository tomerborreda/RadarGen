"""RCS (Radar Cross Section) map creation and conversion functions."""

import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors


#### Conversion functions ####

def rcs_map_to_image(rcs_map: np.ndarray, rcs_min: float, rcs_max: float) -> tuple[Image.Image, np.ndarray]:
    """Convert a RCS map to image representation.

    Returns both a PIL Image for visualization and a normalized float array
    for further processing.

    Args:
        rcs_map: RCS map as (H, W) array with values in [rcs_min, rcs_max]
        rcs_min: Minimum RCS value in dBsm
        rcs_max: Maximum RCS value in dBsm

    Returns:
        Tuple of:
        - PIL Image (uint8 RGB)
        - Normalized float array (H, W, 3) in range [0, 255]

    Raises:
        ValueError: If input array is empty
    """
    if rcs_map.size == 0:
        raise ValueError("RCS map is empty")

    # Normalize to 0-1 range using RCS bounds
    image = (rcs_map - rcs_min) / (rcs_max - rcs_min)
    assert np.all(image >= 0.0) and np.all(image <= 1.0), f"RCS values are out of bounds: [{rcs_map.min()}, {rcs_map.max()}]. Image values are not in the range [0, 1]"
    # Scale to 0-255
    image = image * 255.0

    # Convert grayscale to RGB (3 channels)
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
        image = np.repeat(image, 3, axis=-1)

    # Return both PIL Image (uint8) and float array for downstream processing
    return Image.fromarray(image.astype(np.uint8)), image


def decoded_to_rcs_map(decoded_image: np.ndarray, rcs_min: float, rcs_max: float) -> np.ndarray:
    """Convert a decoded RGB image back to a RCS map.

    This is approximately the inverse of rcs_map_to_image().

    Args:
        decoded_image: RGB image as (H, W, 3) array
        rcs_min: Minimum RCS value in dBsm
        rcs_max: Maximum RCS value in dBsm

    Returns:
        RCS map as (H, W) array with values in [rcs_min, rcs_max]

    Raises:
        ValueError: If image is empty
    """
    if decoded_image.size == 0:
        raise ValueError("Decoded image is empty")

    # Average over RGB channels to get grayscale
    image = np.mean(decoded_image, axis=-1)

    # Denormalize from 0-255 to RCS range
    rcs_map = ((image / 255.0) * (rcs_max - rcs_min)) + rcs_min

    return rcs_map


#### Map creation functions ####

def create_raw_rcs_map(
        locations: np.ndarray,
        rcs_values: np.ndarray,
        image_size: int,
        rcs_min: float,
        rcs_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a sparse RCS map from point locations.

    For pixels with multiple points, keeps the maximum RCS value.
    Uses bottom-left origin coordinate system (y increases upward).

    Args:
        locations: Point locations as (2, N) array where rows are [x, y]
        rcs_values: RCS values as (N,) array in range [rcs_min, rcs_max]
        image_size: Size of output square map (both height and width)
        rcs_min: Minimum RCS value in dBsm
        rcs_max: Maximum RCS value in dBsm

    Returns:
        Tuple of:
        - rcs_map: Sparse RCS map as (height, width) array
        - rcs_map_mask: Boolean mask as (height, width) where True indicates
                       a point exists at that pixel

    Raises:
        ValueError: If input arrays have incorrect shapes, mismatched sizes,
                   or RCS values are out of bounds
    """
    # Validate inputs
    if locations.ndim != 2 or locations.shape[0] != 2:
        raise ValueError(f"Locations must have shape (2, N), got {locations.shape}")
    if rcs_values.ndim != 1:
        raise ValueError(f"RCS values must be 1D, got shape {rcs_values.shape}")
    if locations.shape[1] != rcs_values.shape[0]:
        raise ValueError(
            f"Number of locations ({locations.shape[1]}) doesn't match "
            f"number of RCS values ({rcs_values.shape[0]})"
        )

    # Check RCS value bounds
    if len(rcs_values) > 0:
        if rcs_values.min() < rcs_min or rcs_values.max() > rcs_max:
            raise ValueError(
                f"RCS values must be in [{rcs_min}, {rcs_max}], "
                f"got range [{rcs_values.min()}, {rcs_values.max()}]"
            )

    height = width = image_size

    # Initialize map with minimum RCS value
    rcs_map = np.full((height, width), rcs_min, dtype=np.float32)
    rcs_map_mask = np.zeros((height, width), dtype=bool)

    if len(rcs_values) > 0:
        # Convert to pixel indices
        x_int = np.floor(locations[0]).astype(np.int32)
        y_int = height - 1 - np.floor(locations[1]).astype(np.int32)

        # Filter valid points within bounds
        valid = (x_int >= 0) & (x_int < width) & (y_int >= 0) & (y_int < height)
        x_valid = x_int[valid]
        y_valid = y_int[valid]
        rcs_valid = rcs_values[valid]

        # Convert 2D indices to 1D linear indices
        linear_idx = np.ravel_multi_index((y_valid, x_valid), (height, width))

        # Use numpy ufunc to handle collisions (keep maximum RCS)
        rcs_map_flat = rcs_map.ravel()
        np.maximum.at(rcs_map_flat, linear_idx, rcs_valid)
        rcs_map = rcs_map_flat.reshape(height, width)

        # Update mask for all points that were placed
        rcs_map_mask.ravel()[linear_idx] = True

    return rcs_map, rcs_map_mask


def create_interpolated_rcs_map(
        rcs_map: np.ndarray,
        rcs_map_mask: np.ndarray,
        rcs_min: float,
        max_interpolation_distance: float = None,
) -> np.ndarray:
    """Fill sparse RCS map using nearest neighbor interpolation.

    Pixels far from any known point are set to rcs_min (background).

    Args:
        rcs_map: Sparse RCS map as (height, width) array
        rcs_map_mask: Boolean mask as (height, width) indicating known points
        rcs_min: Minimum RCS value used as background fill (in dBsm)
        max_interpolation_distance: Maximum distance in pixels for interpolation.
                                   Points farther than this are set to rcs_min.
                                   Default: height / 10.0

    Returns:
        Interpolated RCS map as (height, width) array

    Raises:
        ValueError: If inputs have mismatched shapes or no known points exist
    """
    # Validate inputs
    if rcs_map_mask.shape != rcs_map.shape:
        raise ValueError("RCS map and mask must have same shape")

    # Check if we have any known points
    if not np.any(rcs_map_mask):
        raise ValueError(
            "Cannot interpolate RCS map: no known points. "
            "This should have been caught upstream in generator.py."
        )

    height, width = rcs_map.shape

    # Set default max distance
    if max_interpolation_distance is None:
        max_interpolation_distance = height / 10.0

    # Create coordinate grids for all pixels
    grid_x, grid_y = np.meshgrid(
        np.arange(width),
        np.arange(height),
        indexing='xy'
    )

    # Extract known point coordinates and values
    known_y, known_x = np.where(rcs_map_mask)
    known_points = np.column_stack((known_x, known_y))
    known_values = rcs_map[rcs_map_mask]

    # Query points: all pixel locations
    query_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    # Find nearest neighbor for each pixel
    nbrs = NearestNeighbors(n_neighbors=1).fit(known_points)
    distances, indices = nbrs.kneighbors(query_points)

    # Assign RCS from nearest neighbor
    interp_rcs_flat = known_values[indices[:, 0]]

    # Set pixels that are too far from any known point to rcs_min
    is_too_far = distances[:, 0] > max_interpolation_distance
    interp_rcs_flat[is_too_far] = rcs_min

    # Validate no NaN values
    if np.any(np.isnan(interp_rcs_flat)):
        raise RuntimeError("Interpolation produced NaN values")

    # Reshape to image dimensions
    interp_rcs = interp_rcs_flat.reshape(height, width)

    return interp_rcs


def create_radar_rcs_map(
        locations: np.ndarray,
        rcs_values: np.ndarray,
        image_size: int,
        rcs_min: float,
        rcs_max: float,
        max_interpolation_distance: float = None,
) -> np.ndarray:
    """Create an interpolated RCS map from 2D point locations.

    Combines sparse RCS map creation with nearest neighbor interpolation to
    produce a dense RCS map.

    Args:
        locations: Point locations as (2, N) array where rows are [x, y]
        rcs_values: RCS values as (N,) array in range [rcs_min, rcs_max]
        image_size: Size of output square map (both height and width)
        rcs_min: Minimum RCS value in dBsm
        rcs_max: Maximum RCS value in dBsm
        max_interpolation_distance: Maximum distance in pixels for interpolation.
                                   Default: image_size / 10.0

    Returns:
        Interpolated RCS map as (image_size, image_size) array

    Raises:
        ValueError: If locations array has incorrect shape or RCS values out of bounds
    """
    # Validate input shape
    if locations.ndim != 2 or locations.shape[0] != 2:
        raise ValueError(f"Locations must have shape (2, N), got {locations.shape}")

    # Create sparse RCS map
    raw_rcs_map, rcs_map_mask = create_raw_rcs_map(locations, rcs_values, image_size, rcs_min=rcs_min, rcs_max=rcs_max)

    # Interpolate to fill gaps
    return create_interpolated_rcs_map(
        raw_rcs_map,
        rcs_map_mask,
        rcs_min=rcs_min,
        max_interpolation_distance=max_interpolation_distance,
    )
