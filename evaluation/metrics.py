"""Metric functions for point cloud evaluation."""

from typing import Dict, Optional

import numpy as np
import torch
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors
from torch import nn

from radargen.core.normalization import NormalizationConfig

from evaluation.box_utils import convert_pcl_to_box_coords, get_points_in_box


# ---------------------------------------------------------------------------
# Chamfer distance
# ---------------------------------------------------------------------------


# From https://gist.github.com/sergeyprokudin/c4bf4059230da8db8256e36524993367
def chamfer_distance(x, y, metric='l2', direction='bi', return_nn=False):
    r"""Chamfer distance between two point clouds.
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default l2
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    return_nn: bool
        return the nearest neighbors
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """
    min_y_to_x = min_x_to_y = None
    if direction == 'y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction == 'x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction == 'bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")

    if return_nn:
        return chamfer_dist, min_y_to_x, min_x_to_y
    return chamfer_dist


# ---------------------------------------------------------------------------
# MMD (Maximum Mean Discrepancy)
# ---------------------------------------------------------------------------


class RBF(nn.Module):

    def __init__(self,
        n_kernels=5,
        mul_factor=2.0,
        bandwidth=None,
        compute_dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if n_kernels < 1:
            raise ValueError(f"n_kernels must be >= 1, got {n_kernels}")
        if mul_factor <= 0:
            raise ValueError(f"mul_factor must be > 0, got {mul_factor}")
        self.register_buffer(
            'bandwidth_multipliers',
            mul_factor ** (torch.arange(n_kernels, dtype=compute_dtype) - n_kernels // 2),
        )
        self.bandwidth = bandwidth
        self.device = device
        if self.device is not None:
            self.bandwidth_multipliers = self.bandwidth_multipliers.to(self.device)
            if isinstance(self.bandwidth, torch.Tensor):
                self.bandwidth = self.bandwidth.to(self.device)

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            if n_samples < 2:
                return torch.tensor(1.0, device=L2_distances.device, dtype=L2_distances.dtype)
            bandwidth = L2_distances.detach().sum() / (n_samples ** 2 - n_samples)
        elif isinstance(self.bandwidth, torch.Tensor):
            bandwidth = self.bandwidth.to(device=L2_distances.device, dtype=L2_distances.dtype)
        else:
            bandwidth = torch.tensor(float(self.bandwidth), device=L2_distances.device, dtype=L2_distances.dtype)

        eps = torch.finfo(L2_distances.dtype).eps
        bandwidth = bandwidth + eps
        if torch.any(bandwidth <= 0):
            raise ValueError(f"RBF bandwidth must be > 0, got {bandwidth}")
        return bandwidth

    def forward(self, X):
        if self.device is not None:
            X = X.to(self.device)
        L2_distances = torch.cdist(X, X) ** 2  # (|X|+|Y|,|X|+|Y|)
        bandwidth_multipliers = self.bandwidth_multipliers.to(device=L2_distances.device, dtype=L2_distances.dtype)
        if torch.any(bandwidth_multipliers <= 0):
            raise ValueError(f"RBF bandwidth multipliers must be > 0, got {bandwidth_multipliers}")
        return torch.exp(
            -L2_distances[None, ...]
            / (self.get_bandwidth(L2_distances) * bandwidth_multipliers)[:, None, None]
        ).sum(dim=0)


# From https://github.com/yiftachbeer/mmd_loss_pytorch/tree/master
class MMDLoss(nn.Module):

    def __init__(
        self,
        kernel: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        compute_dtype: torch.dtype = torch.float64,
        negative_tolerance: float = 1e-10,
    ):
        super().__init__()
        self.kernel = kernel if kernel is not None else RBF(compute_dtype=compute_dtype, device=device)
        self.device = device
        self.compute_dtype = compute_dtype
        if negative_tolerance < 0:
            raise ValueError(f"negative_tolerance must be >= 0, got {negative_tolerance}")
        self.negative_tolerance = negative_tolerance
        if self.device is not None:
            self.kernel = self.kernel.to(self.device)

    def forward(self, X, Y):
        if self.device is not None:
            X = X.to(self.device)
            Y = Y.to(self.device)

        if torch.isnan(X).any() or torch.isinf(X).any():
            raise ValueError(f"X data contains NaN/Inf {X}")
        if torch.isnan(Y).any() or torch.isinf(Y).any():
            raise ValueError(f"Y data contains NaN/Inf {Y}")

        X_size = X.shape[0]
        Y_size = Y.shape[0]
        if X_size == 0 and Y_size == 0:
            # MMD of two empty sets is 0
            return torch.tensor(0.0, device=X.device, dtype=self.compute_dtype)
        if X_size == 0 or Y_size == 0:
            # MMD between a set and an empty set is not well-defined
            # by this calculation. Raising an error is safer.
            raise ValueError(f"MMDLoss: Input tensors must not be empty. X_size: {X_size}, Y_size: {Y_size}")

        X_kernel = X.to(dtype=self.compute_dtype)
        Y_kernel = Y.to(dtype=self.compute_dtype)
        K = self.kernel(torch.vstack([X_kernel, Y_kernel]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        result = XX - 2 * XY + YY
        if torch.isnan(result):
            raise ValueError(f"MMD is NaN. X_size: {X_size}, Y_size: {Y_size}")
        if torch.isinf(result):
            raise ValueError("MMD is Inf")
        if result < 0:
            raise ValueError(f"MMD is negative: {result.item()}")
        return result


# ---------------------------------------------------------------------------
# F1 / IoU for point clouds
# ---------------------------------------------------------------------------


def pcl_f1_iou_np(synthetic_pcl: np.ndarray, gt_pcl: np.ndarray, th: float = 0.5, suffix: str = ''):
    """F1 and IoU between two point clouds based on nearest-neighbour distances.

    Args:
        synthetic_pcl: (N, D) array - only spatial coordinates should be passed.
        gt_pcl: (M, D) array.
        th: Distance threshold for a *close* point.
        suffix: String appended to the returned keys.

    Returns:
        dict with ``f1<suffix>`` and ``iou<suffix>``.
    """
    def _metrics_pointcloud(pred, target, th):
        P = np.mean(pred < th)
        R = np.mean(target < th)
        if (P < 1e-3) and (R < 1e-3):
            return P, P
        f = 2 * P * R / (P + R)
        iou = P * R / (P + R - (P * R))
        return f, iou

    cd_dist, pred_nn, target_nn = chamfer_distance(synthetic_pcl, gt_pcl, direction='bi', return_nn=True)
    pred_nn, target_nn = np.sqrt(pred_nn), np.sqrt(target_nn)
    f1, iou = _metrics_pointcloud(pred_nn, target_nn, th=th)
    return {
        f"f1{suffix}": f1,
        f"iou{suffix}": iou,
    }


# ---------------------------------------------------------------------------
# Density similarity
# ---------------------------------------------------------------------------


def n_points_similarity_np(synthetic_pcl: np.ndarray, gt_pcl: np.ndarray) -> float:
    """Ratio min(N, M) / max(N, M). Returns 1.0 for two empty sets, 0.0 if only one is empty."""
    n_syn = synthetic_pcl.shape[0]
    n_gt = gt_pcl.shape[0]
    if n_syn == 0 and n_gt == 0:
        return 1.0
    if n_syn == 0 or n_gt == 0:
        return 0.0
    return min(n_syn, n_gt) / max(n_syn, n_gt)


# ---------------------------------------------------------------------------
# Distance + attribute matching
# ---------------------------------------------------------------------------


def distance_attr_recall_precision_f1_hungarian(gt_pcl: torch.Tensor, synthetic_pcl: torch.Tensor, distance_threshold: float = 1.0, rcs_threshold: float = 8.0, doppler_threshold: float = 2.5) -> dict:
    """
    Calculates Recall, Precision, and F1 score for point cloud matching
    using the Hungarian algorithm (linear_sum_assignment) for optimal
    one-to-one matching.

    Args:
        gt_pcl (torch.Tensor): Ground truth point cloud, shape (M, 4) [x, y, rcs, doppler]
        synthetic_pcl (torch.Tensor): Synthetic point cloud, shape (N, 4) [x, y, rcs, doppler]
        distance_threshold (float): Max Euclidean distance for a match.
        rcs_threshold (float): Max absolute difference for RCS for a match.
        doppler_threshold (float): Max absolute difference for Doppler for a match.

    Returns:
        dict: A dictionary containing 'recall', 'precision', and 'f1' scores.
    """
    
    # --- 1. Setup and Edge Case Handling ---
    M = gt_pcl.shape[0] # Number of ground truth points
    N = synthetic_pcl.shape[0] # Number of synthetic points
    eps = 1e-10 # Epsilon for safe division
    device = gt_pcl.device

    # Handle empty point clouds
    # TODO: Raise an error if the point clouds are empty?
    # We don't actually get to this part of the code due to previous checks. These values are undefined when the point clouds are empty.
    if M == 0 and N == 0:
        return {'recall': 1.0, 'precision': 1.0, 'f1': 1.0}
    if M == 0:
        return {'recall': 1.0, 'precision': 0.0, 'f1': 0.0}
    if N == 0:
        return {'recall': 0.0, 'precision': 1.0, 'f1': 0.0}

    # --- 2. Separate Coordinates and Attributes ---
    gt_xy = gt_pcl[:, :2]
    syn_xy = synthetic_pcl[:, :2]
    gt_attrs = gt_pcl[:, 2:]
    syn_attrs = synthetic_pcl[:, 2:]

    # --- 3. Calculate Pairwise Match Conditions ---
    dist_matrix = torch.cdist(gt_xy, syn_xy, p=2.0) # (M, N)
    dist_matches = dist_matrix <= distance_threshold

    attr_diffs = torch.abs(gt_attrs.unsqueeze(1) - syn_attrs.unsqueeze(0))
    rcs_matches = attr_diffs[..., 0] <= rcs_threshold
    doppler_matches = attr_diffs[..., 1] <= doppler_threshold

    # --- 4. Combine Matches ---
    valid_matches = dist_matches & rcs_matches & doppler_matches # (M, N)

    # --- 5. Create Cost Matrix ---
    # Cost is the distance, but only for valid matches.
    cost_matrix = dist_matrix.clone()

    # Use a large finite cost, much larger than any possible valid distance.
    # This ensures the algorithm can always find a solution.
    invalid_cost = distance_threshold + 1e5  # Or any number safely larger than max distance
    cost_matrix[~valid_matches] = invalid_cost

    # --- 6. Run Hungarian Algorithm ---
    # Convert to numpy for scipy
    cost_matrix_np = cost_matrix.cpu().numpy()
    
    # Run the assignment
    gt_indices, syn_indices = linear_sum_assignment(cost_matrix_np)
    
    # --- 7. Calculate TP, FP, FN ---
    
    # The algorithm finds the best matching, but we must check if 
    # the chosen pairs are *actually valid* (i.e., cost wasn't our large 'invalid_cost').
    # We use the original `valid_matches` tensor for this check.
    
    gt_indices_tensor = torch.from_numpy(gt_indices).to(device)
    syn_indices_tensor = torch.from_numpy(syn_indices).to(device)

    # Check which of the assigned pairs are valid
    matched_mask = valid_matches[gt_indices_tensor, syn_indices_tensor]
    
    # TP is the count of *valid* pairs found by the optimal assignment
    TP = matched_mask.sum().float()
    
    # FN = All GT points - Matched GT points
    FN = M - TP
    
    # FP = All Synthetic points - Matched Synthetic points
    FP = N - TP

    # --- 8. Calculate Final Metrics ---
    recall = TP / (TP + FN + eps)
    precision = TP / (TP + FP + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)

    return {'recall': recall.item(), 'precision': precision.item(), 'f1': f1.item()}


# ---------------------------------------------------------------------------
# Normalization helper
# ---------------------------------------------------------------------------


def normalize_pcl_for_chamfer(pcl: torch.Tensor, norm: NormalizationConfig) -> torch.Tensor:
    """Normalize a (N, 4) point cloud [x, y, rcs, doppler] to [0, 1] using *norm*.

    This replaces the previously hardcoded normalization constants.
    """
    cr = norm.coordinate_range
    xy = (pcl[:, :2] - (-cr)) / (2 * cr)
    rcs = (pcl[:, 2:3] - norm.rcs_min) / (norm.rcs_max - norm.rcs_min)
    doppler = (pcl[:, 3:4] - norm.doppler_min) / (norm.doppler_max - norm.doppler_min)
    return torch.cat([xy, rcs, doppler], dim=1)


# ---------------------------------------------------------------------------
# Per-box metrics
# ---------------------------------------------------------------------------


def compute_box_metrics_and_stats(
    synthetic_pcl: np.ndarray,
    gt_pcl: np.ndarray,
    bounding_boxes: list,
    device: torch.device,
    norm_config: NormalizationConfig,
) -> dict:
    """Compute metrics and box statistics within 2D bounding boxes.

    ``bounding_boxes`` are duck-typed: each element must expose
    ``.bottom_corners()`` returning (3, 4), ``.name``, ``.center``,
    ``.rotation_matrix``, and ``.wlh``.

    Args:
        synthetic_pcl: (N, 4) array [x, y, rcs, doppler].
        gt_pcl: (M, 4) array [x, y, rcs, doppler].
        bounding_boxes: List of box objects.
        device: Torch device.
        norm_config: ``NormalizationConfig`` for the active dataset.

    Returns:
        dict of per-box metric lists and count statistics.
    """
    chamfer_distance_xy = []
    chamfer_distance_all_norm = []
    num_points_similarity = []
    invariant_pcl_gt = defaultdict(list)
    invariant_pcl_syn = defaultdict(list)

    n_hits = 0.0
    n_fn_boxes = 0.0
    n_fp_boxes = 0.0

    for box in bounding_boxes:
        bottom_corners = box.bottom_corners()
        synthetic_pcl_in_box, synthetic_pcl_in_box_mask = get_points_in_box(synthetic_pcl, bottom_corners)
        gt_pcl_in_box, gt_pcl_in_box_mask = get_points_in_box(gt_pcl, bottom_corners)

        invariant_pcl_gt[box.name].append(convert_pcl_to_box_coords(gt_pcl_in_box, box)[0])
        invariant_pcl_syn[box.name].append(convert_pcl_to_box_coords(synthetic_pcl_in_box, box)[0])
        num_points_similarity.append(n_points_similarity_np(synthetic_pcl_in_box, gt_pcl_in_box))

        has_syn_points = np.any(synthetic_pcl_in_box_mask)
        has_gt_points = np.any(gt_pcl_in_box_mask)

        if has_gt_points:
            if has_syn_points:
                n_hits += 1

                syn_t = torch.from_numpy(synthetic_pcl_in_box).float().to(device)
                gt_t = torch.from_numpy(gt_pcl_in_box).float().to(device)

                cd_xy = 0.5 * chamfer_distance(
                    syn_t[:, :2].cpu().numpy(), gt_t[:, :2].cpu().numpy(), direction='bi',
                )
                chamfer_distance_xy.append(cd_xy)

                syn_norm = normalize_pcl_for_chamfer(syn_t, norm_config)
                gt_norm = normalize_pcl_for_chamfer(gt_t, norm_config)
                cd_norm = 0.5 * chamfer_distance(
                    syn_norm.cpu().numpy(), gt_norm.cpu().numpy(), direction='bi',
                )
                chamfer_distance_all_norm.append(cd_norm)
            else:
                n_fn_boxes += 1
        else:
            if has_syn_points:
                n_fp_boxes += 1

    return {
        "per_box_chamfer_distance_xy": chamfer_distance_xy,
        "per_box_chamfer_distance_all_norm": chamfer_distance_all_norm,
        "per_box_num_points_similarity": num_points_similarity,
        "per_box_invariant_pcl_gt": invariant_pcl_gt,
        "per_box_invariant_pcl_syn": invariant_pcl_syn,
        "n_hits": n_hits,
        "n_fn_boxes": n_fn_boxes,
        "n_fp_boxes": n_fp_boxes,
    }


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------


def compute_per_sample_metrics(
    synthetic_pcl: np.ndarray,
    gt_pcl: np.ndarray,
    norm_config: NormalizationConfig,
    mmd_loss: MMDLoss,
    device: torch.device,
) -> dict:
    """Compute all per-sample (non-box) metrics.

    Args:
        synthetic_pcl: (N, 4) array [x, y, rcs, doppler].
        gt_pcl: (M, 4) array [x, y, rcs, doppler].
        norm_config: ``NormalizationConfig`` for normalised chamfer distance.
        mmd_loss: Pre-initialised ``MMDLoss``.
        device: Torch device.

    Returns:
        dict of scalar metric values.
    """
    metrics: Dict[str, float] = {}

    syn_torch = torch.from_numpy(synthetic_pcl).float().to(device)
    gt_torch = torch.from_numpy(gt_pcl).float().to(device)

    syn_xy = synthetic_pcl[:, :2]
    gt_xy = gt_pcl[:, :2]

    # IoU
    metrics['iou_th1'] = pcl_f1_iou_np(syn_xy, gt_xy, th=1.0, suffix='_th1')['iou_th1']

    # Chamfer distance (xy only)
    cd_xy = 0.5 * chamfer_distance(syn_xy, gt_xy, direction='bi')
    metrics['chamfer_distance_xy'] = cd_xy

    # Chamfer distance (all attributes, normalised)
    syn_norm = normalize_pcl_for_chamfer(syn_torch, norm_config)
    gt_norm = normalize_pcl_for_chamfer(gt_torch, norm_config)
    cd_norm = 0.5 * chamfer_distance(syn_norm.cpu().numpy(), gt_norm.cpu().numpy(), direction='bi')
    metrics['chamfer_distance_all_norm'] = cd_norm

    # Distance + attribute matching
    res = distance_attr_recall_precision_f1_hungarian(gt_torch, syn_torch)
    metrics['distance_attr_recall'] = res['recall']
    metrics['distance_attr_precision'] = res['precision']
    metrics['distance_attr_f1'] = res['f1']

    # MMD
    metrics['pcl_mmd_xy'] = mmd_loss(syn_torch[:, :2], gt_torch[:, :2]).item()
    metrics['pcl_mmd_rcs'] = mmd_loss(syn_torch[:, 2:3], gt_torch[:, 2:3]).item()
    metrics['pcl_mmd_doppler'] = mmd_loss(syn_torch[:, 3:4], gt_torch[:, 3:4]).item()

    return metrics


def compute_all_metrics(
    synthetic_pcl: np.ndarray,
    gt_pcl: np.ndarray,
    bounding_boxes: list,
    norm_config: NormalizationConfig,
    mmd_loss: MMDLoss,
    device: torch.device,
) -> dict:
    """Compute per-sample metrics + per-box metrics in one call.

    Args:
        synthetic_pcl: (N, 4) array [x, y, rcs, doppler].
        gt_pcl: (M, 4) array [x, y, rcs, doppler].
        bounding_boxes: List of box objects (duck-typed).
        norm_config: ``NormalizationConfig`` for the active dataset.
        mmd_loss: Pre-initialised ``MMDLoss``.
        device: Torch device.

    Returns:
        dict merging per-sample and per-box metrics.
    """
    metrics = compute_per_sample_metrics(synthetic_pcl, gt_pcl, norm_config, mmd_loss, device)
    box_metrics = compute_box_metrics_and_stats(
        synthetic_pcl, gt_pcl, bounding_boxes, device, norm_config,
    )
    metrics.update(box_metrics)
    return metrics
