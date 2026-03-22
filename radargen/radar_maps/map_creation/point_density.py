"""Point density map creation and conversion functions."""

import numpy as np
import torch
from PIL import Image


#### Conversion functions ####

def pd_map_to_image(pd_map: np.ndarray) -> tuple[Image.Image, np.ndarray]:
    """Convert a point density map to image representation.

    Args:
        pd_map: Point density map as (H, W) array

    Returns:
        Tuple of (PIL Image, normalized numpy array as uint8)

    Raises:
        ValueError: If input array is empty or all zeros
    """
    if pd_map.size == 0 or pd_map.max() == 0:
        raise ValueError("Point density map is empty or all zeros")

    # Normalize to 0-1 range
    image = pd_map / pd_map.max()
    # Scale to 0-255
    image = image * 255.0

    # Convert grayscale to RGB (3 channels)
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
        image = np.repeat(image, 3, axis=-1)

    return Image.fromarray(image.astype(np.uint8)), image


def decoded_to_pd_map(decoded_image: np.ndarray) -> np.ndarray:
    """Convert a decoded RGB image back to a point density map.

    This is approximately the inverse of pd_map_to_image().

    Args:
        decoded_image: RGB image as (H, W, 3) array

    Returns:
        Point density map as (H, W) array normalized to sum to 1

    Raises:
        ValueError: If image is empty or all zeros
    """
    # Average over RGB channels to get grayscale
    image = np.mean(decoded_image, axis=-1)

    if image.size == 0 or image.max() == 0:
        raise ValueError("Decoded image is empty or all zeros")

    # Normalize to 0-255 range
    image = 255 * image / image.max()

    # Normalize to probability distribution (sum to 1)
    image_sum = image.sum()
    if image_sum == 0:
        raise ValueError("Image sum is zero, cannot normalize")

    return image / image_sum


#### Map creation functions ####

def point_cloud_to_point_density_map(
        points: np.ndarray,
        width: int,
        height: int,
        device: torch.device,
        sigma_range: tuple[float, float] = (2.0, 2.0),
        sigma_threshold: float = 3.0,
) -> np.ndarray:
    """Generate a point density map using Gaussian splatting with GPU acceleration.

    Each point is represented as a 2D Gaussian blob, and all Gaussians are summed
    to create a probability density map. Uses a bottom-left origin coordinate system.

    Args:
        points: 2D point locations as (2, N) array where rows are [x, y]
        width: Width of output image in pixels
        height: Height of output image in pixels
        device: PyTorch device for GPU computation
        sigma_range: Gaussian standard deviations as (sigma_x, sigma_y) tuple
        sigma_threshold: Distance threshold in standard deviations for truncating
                        Gaussians (default 3.0 = 99.7% of distribution)

    Returns:
        Point density map as (height, width) array normalized to sum to 1

    Raises:
        ValueError: If points array has incorrect shape or is empty
    """
    # Validate input shape
    if points.ndim != 2 or points.shape[0] != 2:
        raise ValueError(f"Points must have shape (2, N), got {points.shape}")

    # Validate non-empty point cloud
    num_points = points.shape[1]
    if num_points == 0:
        raise ValueError(
            "Cannot create point density map from empty point cloud. "
            "This should have been caught upstream in generator.py. "
            "Check that validation is enabled."
        )

    # Move points to GPU
    points_gpu = torch.from_numpy(points).float().to(device)

    # Create coordinate grids with bottom-left origin
    # Y-axis increases upward (image row 0 = y_max, image row H-1 = y_min)
    x_grid, y_grid = torch.meshgrid(
        torch.arange(width, device=device),
        torch.arange(height - 1, -1, -1, device=device),  # Reversed y-axis
        indexing='xy'
    )

    # Reshape grids for broadcasting: (H*W, 1)
    x_grid_flat = x_grid.reshape(-1, 1)
    y_grid_flat = y_grid.reshape(-1, 1)

    # Extract sigma values
    sigma_x, sigma_y = sigma_range

    # Compute distances from each grid point to all points: (H*W, N)
    dx = x_grid_flat - points_gpu[0]
    dy = y_grid_flat - points_gpu[1]

    # Compute normalized squared distances
    dist_sq_normalized = (dx**2 / sigma_x**2) + (dy**2 / sigma_y**2)

    # Calculate 2D Gaussian values
    exponent = -0.5 * dist_sq_normalized
    coeff = 1.0 / (2 * torch.pi * sigma_x * sigma_y)
    gaussians = coeff * torch.exp(exponent)

    # Apply distance threshold mask to truncate far-away values
    # This saves computation and reduces noise
    distance_mask = torch.any(
        dist_sq_normalized < sigma_threshold**2,
        dim=1,
        keepdim=True
    )
    gaussians = gaussians * distance_mask

    # Sum Gaussians from all points and reshape to image
    density_map = torch.sum(gaussians, dim=1).reshape(height, width)

    # Normalize to probability distribution (sum to 1)
    density_sum = density_map.sum()
    if density_sum == 0:
        raise RuntimeError(
            "Density map sum is zero. This may occur if all points are outside "
            "the image bounds or sigma_threshold is too small."
        )

    density_map = density_map / density_sum

    # Move back to CPU and convert to numpy
    return density_map.detach().cpu().numpy()


def create_radar_pd_map(
        locations: np.ndarray,
        image_size: int,
        device: torch.device,
        sigma_range: tuple[float, float] = (2.0, 2.0),
        sigma_threshold: float = 3.0,
) -> np.ndarray:
    """Create a point density map from 2D point locations using Gaussian splatting.

    Args:
        locations: Point locations as (2, N) array where rows are [x, y]
        image_size: Size of output square map (both height and width)
        device: PyTorch device for GPU computation
        sigma_range: Gaussian standard deviations as (sigma_x, sigma_y) tuple
        sigma_threshold: Distance threshold in sigmas for truncating Gaussians

    Returns:
        Point density map as (image_size, image_size) array normalized to sum to 1

    Raises:
        ValueError: If locations array has incorrect shape or is empty
    """
    # Validate input shape
    if locations.ndim != 2 or locations.shape[0] != 2:
        raise ValueError(f"Locations must have shape (2, N), got {locations.shape}")

    return point_cloud_to_point_density_map(
        points=locations,
        width=image_size,
        height=image_size,
        device=device,
        sigma_range=sigma_range,
        sigma_threshold=sigma_threshold,
    )
