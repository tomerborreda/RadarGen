import numpy as np
import torch
import torch.nn.functional as F
import utils3d
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from unidepth.models import UniDepthV2
from uniflowmatch.models.ufm import UniFlowMatchConfidence

from radargen.radar_maps import doppler_map_to_image


PALETTE_CITYSCAPES = np.array([
    [128,  64, 128],  # road
    [244,  35, 232],  # sidewalk
    [ 70,  70,  70],  # building
    [70,  70,  70],  # wall
    [190, 153, 153],  # fence
    [153, 153, 153],  # pole
    [250, 170,  30],  # traffic light
    [220, 220,   0],  # traffic sign
    [107, 142,  35],  # vegetation
    [107, 142,  35],  # terrain
    [ 70, 130, 180],  # sky
    [220,  20,  60],  # person
    [255,   0,   0],  # rider
    [  0,   0, 142],  # car
    [  0,   0,  70],  # truck
    [  0,  60, 100],  # bus
    [  0,  80, 100],  # train
    [  0,   0, 230],  # motorcycle
    [119,  11,  32],  # bicycle
], dtype=np.uint8)


def load_models(device: torch.device):
    depth_model = UniDepthV2.from_pretrained(f"lpiccinelli/unidepth-v2-vitl14").to(device).eval()
    depth_model.resolution_level = 9
    depth_model.interpolation_mode = "bilinear"

    segmentation_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
    segmentation_model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic").to(device).eval() # For less vram use torch_dtype=torch.float16

    flow_model = UniFlowMatchConfidence.from_pretrained("infinity1096/UFM-Base-980").to(device).eval()

    return depth_model, segmentation_model, segmentation_processor, flow_model


@torch.inference_mode()
def get_depth(depth_model, input_image: np.ndarray, camera_intrinsics: np.ndarray, device: torch.device) -> dict:
    input_image_depth = torch.tensor(input_image, dtype=torch.float32, device=device).permute(2, 0, 1)
    camera_intrinsics = torch.tensor(camera_intrinsics, dtype=torch.float32, device=device)
    depth_output = depth_model.infer(input_image_depth, camera=camera_intrinsics)
    depth_output = {'points': depth_output['points'][0].permute(1, 2, 0), 'depth': depth_output['depth'][0,0]}
    return depth_output


@torch.inference_mode()
def get_depth_batched(
    depth_model,
    images: list[np.ndarray],
    intrinsics_list: list[np.ndarray],
    device: torch.device
) -> list[dict]:
    """Process multiple camera images in a single batch for depth estimation.

    IMPORTANT: Only works when all cameras share the same intrinsics.
    The caller should verify this before calling this function.

    Args:
        depth_model: UniDepthV2 model
        images: List of RGB images, each (H, W, 3)
        intrinsics_list: List of 3x3 intrinsic matrices (must all be identical)
        device: PyTorch device

    Returns:
        List of depth output dicts, each with 'points' and 'depth' keys
    """
    # Stack all images into a batch
    images_batch = torch.stack([
        torch.tensor(img, dtype=torch.float32, device=device).permute(2, 0, 1)
        for img in images
    ])  # Shape: (B, 3, H, W)

    # Use the first intrinsics since they're all the same
    # UniDepth expects a single (3, 3) matrix for batched images with same camera
    single_intrinsics = torch.tensor(intrinsics_list[0], dtype=torch.float32, device=device)

    # Single batched forward pass
    depth_output = depth_model.infer(images_batch, camera=single_intrinsics)

    # Unpack batch results
    return [
        {
            'points': depth_output['points'][i].permute(1, 2, 0),
            'depth': depth_output['depth'][i, 0]
        }
        for i in range(len(images))
    ]


def segment_image(segmentation_model, segmentation_processor, camera_image: np.ndarray, device: torch.device) -> np.ndarray:
    img = Image.fromarray(camera_image.astype(np.uint8)).convert("RGB")
    h, w = img.size[1], img.size[0]   # (height, width)
    inputs = segmentation_processor(
        images=img,
        do_resize=False,          # keep your native resolution
        return_tensors="pt",
    )
    for k in inputs:
        inputs[k] = inputs[k].to(device)
    with torch.no_grad():
        outputs = segmentation_model(**inputs)
    # Post-process to get class IDs per pixel at original size
    seg = segmentation_processor.post_process_semantic_segmentation(
        outputs=outputs, target_sizes=[(h, w)]
    )[0]  # (H,W) LongTensor with values in [0, num_labels-1]
    seg_np = seg.cpu().numpy()
    return PALETTE_CITYSCAPES[seg_np]  # (H,W,3)


@torch.inference_mode()
def segment_images_batched(
    segmentation_model,
    segmentation_processor,
    images: list[np.ndarray],
    device: torch.device
) -> list[np.ndarray]:
    """Process multiple images through segmentation model in batch.

    Args:
        segmentation_model: Mask2Former model
        segmentation_processor: Mask2Former processor
        images: List of RGB images, each (H, W, 3)
        device: PyTorch device

    Returns:
        List of segmented images, each (H, W, 3) with RGB palette
    """
    # Convert to PIL images
    pil_images = [Image.fromarray(img.astype(np.uint8)).convert("RGB") for img in images]
    image_sizes = [(img.size[1], img.size[0]) for img in pil_images]  # (H, W) for each

    # Process all images together
    inputs = segmentation_processor(
        images=pil_images,
        do_resize=False,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Batched inference
    outputs = segmentation_model(**inputs)

    # Post-process each result
    seg_results = segmentation_processor.post_process_semantic_segmentation(
        outputs=outputs,
        target_sizes=image_sizes
    )

    # Apply palette to each segmentation map
    return [PALETTE_CITYSCAPES[seg.cpu().numpy()] for seg in seg_results]


def radial_velocity_from_flow(
        flow_model,
        input_image_t0: np.ndarray,
        depth_output_t0: dict,
        input_image_t1: np.ndarray,
        depth_output_t1: dict,
        device: torch.device,
        dt: float,
        doppler_min: float,
        doppler_max: float,
        covisibility_threshold: float = 0.5,
    ):
    """
    Compute radial velocity from optical flow between two images.

    Args:
        input_image_t0: Source image at time t0 (numpy array)
        depth_output_t0: Depth output dict containing 'points' key with shape (H, W, 3)
        input_image_t1: Target image at time t1 (numpy array)
        depth_output_t1: Depth output dict containing 'points' key with shape (H, W, 3)
        flow_model: Optical flow model for predicting correspondences
        device: PyTorch device
        dt: Time delta between the two frames in seconds
        doppler_min: Minimum radial velocity bound in m/s
        doppler_max: Maximum radial velocity bound in m/s
        covisibility_threshold: Threshold for covisibility mask

    Returns:
        tuple: (radial_velocity_map, colored_image)
            - radial_velocity_map: 2D numpy array of radial velocities (m/s)
            - colored_image: RGB image visualization of radial velocity
    """
    with torch.no_grad():
        # Compute optical flow from target (t1) to source (t0)
        result = flow_model.predict_correspondences_batched(
            source_image=torch.from_numpy(input_image_t0).to(device).to(torch.uint8),
            target_image=torch.from_numpy(input_image_t1).to(device).to(torch.uint8),
        )
        flow = result.flow.flow_output[0].cpu().numpy()
        covisibility = result.covisibility.mask[0].cpu().numpy()
    
    # Extract point maps
    source_point_map = depth_output_t0['points']  # (H, W, 3)
    target_point_map = depth_output_t1['points']  # (H, W, 3)
    
    # Convert to numpy for processing
    source_point_map_np = source_point_map.cpu().numpy()
    target_point_map_np = target_point_map.cpu().numpy()
    
    H, W, C = source_point_map_np.shape
    assert C == 3, f"Expected 3 channels for 3D points, got {C}"
    
    # Create coordinate grid for source image
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    
    # Apply flow displacements to get target coordinates (target→source flow)
    # Flow comes as (2, H, W), need to transpose to (H, W, 2)
    flow_transposed = flow.transpose(1, 2, 0)
    flow_x, flow_y = flow_transposed[..., 0], flow_transposed[..., 1]
    x_new = x + flow_x
    y_new = y + flow_y
    
    # Normalize coordinates for grid_sample (PyTorch expects [-1, 1] range)
    x_norm = (x_new / (W - 1)) * 2 - 1
    y_norm = (y_new / (H - 1)) * 2 - 1
    
    # Stack coordinates for grid_sample
    grid = np.stack([x_norm, y_norm], axis=-1)
    
    # Convert to torch tensors
    target_points_tensor = target_point_map
    grid_tensor = torch.from_numpy(grid).float().to(target_points_tensor.device)
    
    # Add batch dimension
    target_points_batch = target_points_tensor.unsqueeze(0)  # (1, H, W, 3)
    grid_batch = grid_tensor.unsqueeze(0)  # (1, H, W, 2)
    
    # Use grid_sample to warp the target points
    # Permute to (B, C, H, W) format for grid_sample
    target_points_permuted = target_points_batch.permute(0, 3, 1, 2)  # (1, 3, H, W)
    
    warped_points_permuted = F.grid_sample(
        target_points_permuted,
        grid_batch,
        mode="bilinear", 
        align_corners=False,
        padding_mode="border",
    )
    
    # Convert back to (B, H, W, C) format
    warped_points_batch = warped_points_permuted.permute(0, 2, 3, 1)  # (1, H, W, 3)
    warped_target_points = warped_points_batch.squeeze(0).cpu().numpy()
    
    # Calculate 3D displacement and velocity
    displacement = warped_target_points - source_point_map_np
    velocity = displacement / dt
    
    # Apply covisibility mask
    velocity[covisibility < covisibility_threshold] = 0.0
    
    # Compute radial velocity
    # Calculate unit radial direction from warped target points
    warped_norms = np.linalg.norm(warped_target_points, axis=-1, keepdims=True)
    # Avoid division by zero
    warped_norms = np.where(warped_norms < 1e-8, 1.0, warped_norms)
    r_hat = warped_target_points / warped_norms
    
    # Project velocity onto radial direction
    radial_velocity = np.sum(velocity * r_hat, axis=-1)
    
    # Apply covisibility mask and velocity bounds
    radial_velocity[covisibility < covisibility_threshold] = 0.0
    radial_velocity[radial_velocity > doppler_max] = 0.0
    radial_velocity[radial_velocity < doppler_min] = 0.0

    # Generate colored output
    # colored_image = doppler_values_to_colors(radial_velocity)
    _, colored_image = doppler_map_to_image(radial_velocity, doppler_min=doppler_min, doppler_max=doppler_max)

    return radial_velocity, colored_image


@torch.inference_mode()
def radial_velocity_from_flow_batched(
    flow_model,
    images_t0: list[np.ndarray],
    depth_outputs_t0: list[dict],
    images_t1: list[np.ndarray],
    depth_outputs_t1: list[dict],
    device: torch.device,
    dts: list[float],
    doppler_min: float,
    doppler_max: float,
    covisibility_threshold: float = 0.5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Compute radial velocity for multiple camera pairs.

    Current limitation: UFM model API processes pairs sequentially.
    This wrapper prepares for future batched flow computation.

    Args:
        flow_model: Optical flow model
        images_t0: List of source images at time t0
        depth_outputs_t0: List of depth outputs at time t0
        images_t1: List of target images at time t1
        depth_outputs_t1: List of depth outputs at time t1
        device: PyTorch device
        dts: Per-camera time deltas in seconds
        doppler_min: Minimum radial velocity bound in m/s
        doppler_max: Maximum radial velocity bound in m/s
        covisibility_threshold: Threshold for covisibility mask

    Returns:
        List of (radial_velocity_map, colored_image) tuples
    """
    results = []
    for img_t0, depth_t0, img_t1, depth_t1, dt in zip(
        images_t0, depth_outputs_t0, images_t1, depth_outputs_t1, dts
    ):
        result = radial_velocity_from_flow(
            flow_model, img_t0, depth_t0, img_t1, depth_t1,
            device, dt, doppler_min, doppler_max, covisibility_threshold,
        )
        results.append(result)
    return results


def get_points_mask(depth_outputs: list[dict], segmented_images: list[np.ndarray]) -> np.ndarray:
    # Preprocess the depth output to get a mask of the points.
    # Remove the edges of objects which reduce noise.
    all_masks_l = []
    for i, depth_output in enumerate(depth_outputs):
        depth = depth_output['depth'].cpu().numpy()
        mask = np.ones_like(depth, dtype=bool)
        edge_tol = 0.06
        
        edge_mask = utils3d.numpy.depth_edge(depth, rtol=edge_tol, mask=mask)
        mask = mask & ~edge_mask

        # Mask out the sky
        mask = mask & (segmented_images[i][:,:,0] != 70) & (segmented_images[i][:,:,1] != 130) & (segmented_images[i][:,:,2] != 180)

        # Mask according to confidence
        # confidence_threshold = np.percentile(output['confidence'][0,0].cpu().numpy(), 60)
        # print(f"Confidence threshold: {confidence_threshold}")
        # mask = mask & (output['confidence'][0,0].cpu().numpy() > confidence_threshold)

        mask = mask.reshape(-1)
        all_masks_l.append(mask)
    return np.concatenate(all_masks_l, axis=0)


def points_to_bev_map(
    points: np.ndarray,
    mask: np.ndarray,
    images: list[np.ndarray],
    original_height: int,
    original_width: int,
    resolution: int,
    coordinate_range: float,
) -> np.ndarray:
    """
    Convert a point cloud to a bird eye view map.
    """
    # Create an empty colored bev map
    bev_map = np.zeros((resolution, resolution, 3), dtype=np.float32)
    
    # Filter out points that are not finite or within the coordinate range
    finite_mask = np.isfinite(points).all(axis=1)
    range_mask = (
        (points[:, 0] >= -coordinate_range) & (points[:, 0] <= coordinate_range) &
        (points[:, 1] >= -coordinate_range) & (points[:, 1] <= coordinate_range)
    )
    mask = mask & finite_mask & range_mask
    
    assert np.any(mask), "There are no valid points to convert to a bev map"

    # Get valid data
    points = points[mask]
    indices = np.where(mask)[0]
    
    # Calculate image and pixel coordinates
    points_per_image = original_height * original_width
    image_indices = indices // points_per_image
    point_indices_in_image = indices % points_per_image
    pixel_rows = point_indices_in_image // original_width
    pixel_cols = point_indices_in_image % original_width
    
    # Filter out points that are not within the bounds of the images
    bounds_mask = (
        (image_indices < len(images)) &
        (pixel_rows < original_height) & (pixel_cols < original_width)
    )
    
    assert np.any(bounds_mask), "There are no points within the bounds of the images to convert to a bev map"

    # Apply bounds filter
    points = points[bounds_mask]
    image_indices = image_indices[bounds_mask]
    pixel_rows = pixel_rows[bounds_mask]
    pixel_cols = pixel_cols[bounds_mask]
    
    # Convert to grid coordinates
    x_step = (2 * coordinate_range) / (resolution - 1)
    y_step = (2 * coordinate_range) / (resolution - 1)
    
    x_indices = np.clip(((points[:, 0] + coordinate_range) / x_step).astype(int), 0, resolution - 1)
    y_indices = np.clip(((points[:, 1] + coordinate_range) / y_step).astype(int), 0, resolution - 1)
    y_indices = resolution - 1 - y_indices
    
    # Extract colors efficiently
    all_colors = np.zeros((len(points), 3), dtype=np.float32)
    for img_idx in np.unique(image_indices):
        img_mask = image_indices == img_idx
        all_colors[img_mask] = images[img_idx][pixel_rows[img_mask], pixel_cols[img_mask]]
    
    # Use 'top first' aggregation method for each cell - keep the point with larger z value
    grid_coords = y_indices * resolution + x_indices
    z_values = points[:, 2]
    
    # Filter out points with z > 5
    z_filter = z_values <= 5.0
    assert np.any(z_filter), "All points have height > 5m"
    
    # Apply z filter to all arrays
    grid_coords = grid_coords[z_filter]
    z_values = z_values[z_filter]
    y_indices = y_indices[z_filter]
    x_indices = x_indices[z_filter]
    all_colors = all_colors[z_filter]
    
    # Sort by grid coordinates, then by z values (descending) to get max z first per cell
    sort_keys = np.lexsort((-z_values, grid_coords))
    
    # Get unique coordinates and their first occurrence (max z)
    unique_coords, unique_indices = np.unique(grid_coords[sort_keys], return_index=True)
    max_indices = sort_keys[unique_indices]
    
    # Assign colors for selected points
    bev_map[y_indices[max_indices], x_indices[max_indices]] = all_colors[max_indices]
    return bev_map.astype(np.uint8)
