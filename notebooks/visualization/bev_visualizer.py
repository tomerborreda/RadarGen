"""BEV visualization utilities for radar point clouds."""

from typing import List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from radargen.core.normalization import NormalizationConfig, default_normalization


def visualize_pcl_on_bev(
    pcl: np.ndarray,
    bev_appearance_map: np.ndarray,
    normalization: Optional[NormalizationConfig] = None,
    doppler_vis_limit: float = 20.0,
    title: str = "Radar PCL on BEV",
    show_colorbar: bool = True,
    show_ego_vehicle: bool = True,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Visualize radar point cloud on top of a BEV appearance map.

    Points are colored by Doppler velocity and sized by RCS.

    Args:
        pcl: Point cloud array with shape (N, 5) containing [x, y, z, rcs, doppler]
        bev_appearance_map: BEV appearance map as numpy array (H, W, 3)
        normalization: Normalization config for coordinate range and RCS limits
        doppler_vis_limit: Limit for Doppler colormap (symmetric around 0)
        title: Plot title
        show_colorbar: Whether to show the Doppler colorbar
        show_ego_vehicle: Whether to show ego vehicle indicator
        ax: Optional matplotlib axes to plot on
        save_path: Optional path to save the figure

    Returns:
        Figure object if ax was not provided, None otherwise
    """
    if normalization is None:
        normalization = default_normalization

    coordinate_range = normalization.coordinate_range

    if len(pcl) == 0:
        print("No points to visualize.")
        return None

    # Extract point cloud attributes
    xy_data = pcl[:, :2]  # x, y
    rcs_data = pcl[:, 3]  # rcs
    doppler_data = pcl[:, 4]  # doppler

    # Visualization parameters
    DOPPLER_MIN, DOPPLER_MAX = -doppler_vis_limit, doppler_vis_limit
    DOPPLER_CMAP = "berlin"
    MIN_SIZE, MAX_SIZE = 1, 75

    point_sizes = np.interp(
        rcs_data,
        [normalization.rcs_min, normalization.rcs_max],
        [MIN_SIZE, MAX_SIZE],
    )

    # Create figure if no axes provided
    created_figure = ax is None
    if created_figure:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=150)
    else:
        fig = ax.get_figure()

    # Display BEV appearance map as background
    extent = [-coordinate_range, coordinate_range, -coordinate_range, coordinate_range]
    ax.imshow(bev_appearance_map, extent=extent, origin="upper", zorder=1)

    # Scatter plot for radar points
    doppler_norm = mcolors.Normalize(vmin=DOPPLER_MIN, vmax=DOPPLER_MAX)

    sc = ax.scatter(
        xy_data[:, 0],
        xy_data[:, 1],
        s=point_sizes,
        c=doppler_data,
        cmap=DOPPLER_CMAP,
        norm=doppler_norm,
        alpha=0.8,
        edgecolors="white",
        linewidths=0.3,
        zorder=10,
    )

    # Add colorbar for Doppler
    if show_colorbar:
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Doppler Velocity (m/s)", fontsize=10)

    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect("equal", "box")
    ax.set_xlim(-coordinate_range, coordinate_range)
    ax.set_ylim(-coordinate_range, coordinate_range)

    # Ego-vehicle visualization
    if show_ego_vehicle:
        car_width = 2.0
        car_length = 4.5

        car_body = plt.Rectangle(
            (-car_width / 2, -car_length / 2),
            car_width,
            car_length,
            facecolor="dimgray",
            edgecolor="black",
            linewidth=1.5,
            zorder=20,
        )
        ax.add_patch(car_body)

        windshield = plt.Rectangle(
            (-car_width / 2 + 0.1, car_length / 4),
            car_width - 0.2,
            car_length / 4,
            facecolor="lightgray",
            edgecolor="black",
            linewidth=0.5,
            zorder=21,
        )
        ax.add_patch(windshield)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    # if created_figure:
    #     return fig
    # return None


def visualize_bev_maps(
    bev_appearance_map: np.ndarray,
    bev_segmentation_map: np.ndarray,
    bev_velocity_map: np.ndarray,
    title: str = "BEV Conditioning Maps",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize the three BEV conditioning maps side by side.

    Args:
        bev_appearance_map: BEV color/appearance map
        bev_segmentation_map: BEV semantic segmentation map
        bev_velocity_map: BEV velocity map
        title: Figure title
        save_path: Optional path to save the figure

    Returns:
        Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(bev_appearance_map)
    axes[0].set_title("BEV Appearance Map")
    axes[0].axis("off")

    axes[1].imshow(bev_segmentation_map)
    axes[1].set_title("BEV Segmentation Map")
    axes[1].axis("off")

    axes[2].imshow(bev_velocity_map)
    axes[2].set_title("BEV Velocity Map")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    # return fig


def visualize_camera_images(
    images: List[np.ndarray],
    camera_names: Optional[List[str]] = None,
    title: str = "Multi-View Camera Images",
    ncols: Optional[int] = None,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize multi-view camera images in a grid layout.

    Args:
        images: List of camera images as numpy arrays (H, W, 3) in RGB format
        camera_names: Optional list of camera names for subplot titles
        title: Figure title
        ncols: Number of columns in the grid (default: auto-computed for roughly square layout)
        figsize: Optional figure size as (width, height). If None, auto-computed based on images
        save_path: Optional path to save the figure

    Returns:
        Figure object
    """
    num_images = len(images)
    if num_images == 0:
        print("No images to visualize.")
        return None

    # Auto-compute grid dimensions if not specified
    if ncols is None:
        # Heuristic: try to make roughly square layout
        ncols = int(np.ceil(np.sqrt(num_images)))
    nrows = int(np.ceil(num_images / ncols))

    # Auto-compute figure size if not specified
    if figsize is None:
        # Base size per subplot, scaled by image aspect ratio
        img_height, img_width = images[0].shape[:2]
        aspect_ratio = img_width / img_height
        subplot_height = 3.0
        subplot_width = subplot_height * aspect_ratio
        figsize = (subplot_width * ncols, subplot_height * nrows)

    # Create figure and subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Handle case of single image (axes is not an array)
    if num_images == 1:
        axes = np.array([axes])

    # Flatten axes array for easier indexing
    axes = np.array(axes).flatten()

    # Plot each camera image
    for i, img in enumerate(images):
        axes[i].imshow(img)
        axes[i].axis("off")

        # Set subplot title if camera names provided
        if camera_names is not None and i < len(camera_names):
            axes[i].set_title(camera_names[i], fontsize=10, fontweight="bold")

    # Hide unused subplots
    for i in range(num_images, len(axes)):
        axes[i].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    # return fig
