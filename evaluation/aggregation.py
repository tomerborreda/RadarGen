"""Metric aggregation across evaluation samples."""

import logging
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch

from evaluation.metrics import MMDLoss, RBF


PER_SAMPLE_METRICS = [
    'chamfer_distance_xy',
    'chamfer_distance_all_norm',
    'distance_attr_recall',
    'distance_attr_precision',
    'distance_attr_f1',
    'pcl_mmd_xy',
    'pcl_mmd_rcs',
    'pcl_mmd_doppler',
    'iou_th1',
]

PER_BOX_METRICS = [
    'per_box_chamfer_distance_xy',
    'per_box_chamfer_distance_all_norm',
    'per_box_num_points_similarity',
]

COUNT_METRICS = [
    'n_hits',
    'n_fn_boxes',
    'n_fp_boxes',
]


# ---- Factory ---------------------------------------------------------------


def create_aggregator() -> dict:
    """Return a fresh aggregator dictionary ready for ``aggregate_metrics``."""
    return {
        'per_sample_scores': defaultdict(list),
        'per_box_scores': defaultdict(list),
        'raw_counts': defaultdict(float),
        'agg_points_per_class': defaultdict(lambda: {'syn': [], 'gt': []}),
    }


# ---- Per-sample accumulation -----------------------------------------------


def aggregate_metrics(
    global_aggregator: dict,
    sample_metrics: dict,
    mmd_report_classes: List[str],
) -> None:
    """Append one sample's metrics to *global_aggregator* (in-place).

    Args:
        global_aggregator: Created via ``create_aggregator()``.
        sample_metrics: dict returned by ``compute_all_metrics()``.
        mmd_report_classes: List of class names for invariant MMD (from config).
    """
    for key in PER_SAMPLE_METRICS:
        if key not in sample_metrics:
            raise ValueError(f"Key {key} not found in sample_metrics")
        global_aggregator['per_sample_scores'][key].append(sample_metrics[key])

    for key in PER_BOX_METRICS:
        if key not in sample_metrics:
            raise ValueError(f"Key {key} not found in sample_metrics")
        global_aggregator['per_box_scores'][key].extend(sample_metrics[key])

    for key in COUNT_METRICS:
        if key not in sample_metrics:
            raise ValueError(f"Key {key} not found in sample_metrics")
        global_aggregator['raw_counts'][key] += sample_metrics[key]

    gt_pcls = sample_metrics.get('per_box_invariant_pcl_gt', {})
    syn_pcls = sample_metrics.get('per_box_invariant_pcl_syn', {})
    all_class_names = set(gt_pcls.keys()) | set(syn_pcls.keys())

    for class_name in all_class_names:
        if class_name not in mmd_report_classes:
            continue
        gt_pcl_list = gt_pcls.get(class_name, [])
        if gt_pcl_list:
            valid = [p for p in gt_pcl_list if p.shape[0] > 0]
            if valid:
                global_aggregator['agg_points_per_class'][class_name]['gt'].append(np.vstack(valid))
        syn_pcl_list = syn_pcls.get(class_name, [])
        if syn_pcl_list:
            valid = [p for p in syn_pcl_list if p.shape[0] > 0]
            if valid:
                global_aggregator['agg_points_per_class'][class_name]['syn'].append(np.vstack(valid))


# ---- Finalisation ----------------------------------------------------------


def _safe_mean_std(values: list) -> tuple:
    """Return (mean, std) from a list, filtering NaN/Inf. Returns (nan, nan) if empty."""
    if not values:
        return np.nan, np.nan
    arr = np.array(values)
    valid = arr[np.isfinite(arr)]
    if valid.shape[0] == 0:
        return np.nan, np.nan
    return float(np.mean(valid)), float(np.std(valid))


def finalize_global_metrics(
    aggregator: dict,
    device: torch.device,
    mmd_report_classes: List[str],
    logger: logging.Logger,
) -> dict:
    """Compute final statistics from accumulated data.

    Args:
        aggregator: Populated aggregator dict.
        device: Torch device for MMD computation.
        mmd_report_classes: Class names for invariant MMD reporting (from config).
        logger: Logger instance.

    Returns:
        dict of final metric values.
    """
    final: Dict[str, float] = {}

    # 1. Per-sample mean / std
    for key, value_list in aggregator['per_sample_scores'].items():
        m, s = _safe_mean_std(value_list)
        final[f'{key}_mean'] = m
        final[f'{key}_std'] = s

    # 2. Per-box mean / std
    for key, value_list in aggregator['per_box_scores'].items():
        logger.info(f"The key '{key}' has {len(value_list)} values")
        m, s = _safe_mean_std(value_list)
        logger.info(f"The key '{key}' has {len([v for v in value_list if np.isfinite(v)])} valid values")
        final[f'{key}_mean'] = m
        final[f'{key}_std'] = s

    # 3. Hit / miss rates
    total_hits = aggregator['raw_counts'].get('n_hits', 0.0)
    total_fn = aggregator['raw_counts'].get('n_fn_boxes', 0.0)
    total_fp = aggregator['raw_counts'].get('n_fp_boxes', 0.0)

    final['total_hits'] = total_hits
    final['total_fn_boxes'] = total_fn
    final['total_fp_boxes'] = total_fp

    total_gt_with = total_hits + total_fn
    if total_gt_with > 0:
        final['hit_rate'] = total_hits / total_gt_with
        final['miss_rate'] = total_fn / total_gt_with
    else:
        final['hit_rate'] = np.nan
        final['miss_rate'] = np.nan

    # 4. Invariant MMD per class
    mmd_loss_compute = MMDLoss(kernel=RBF(device=device), device=device)

    for class_name in mmd_report_classes:
        key_xy = f'invariant_mmd_xy_{class_name}'
        key_rcs = f'invariant_mmd_rcs_{class_name}'
        key_doppler = f'invariant_mmd_doppler_{class_name}'

        final[key_xy] = np.nan
        final[key_rcs] = np.nan
        final[key_doppler] = np.nan

        if class_name not in aggregator['agg_points_per_class']:
            continue

        gt_list = aggregator['agg_points_per_class'][class_name].get('gt', [])
        syn_list = aggregator['agg_points_per_class'][class_name].get('syn', [])
        if not gt_list or not syn_list:
            continue

        # Global invariant MMD (subsample to avoid OOM)
        # gt_all = np.vstack(gt_list[::3])
        # syn_all = np.vstack(syn_list[::3])
        gt_all = np.vstack(gt_list[::3][:200])
        syn_all = np.vstack(syn_list[::3][:200])
        if gt_all.shape[0] == 0 or syn_all.shape[0] == 0:
            continue
        gt_torch = torch.from_numpy(gt_all).float().to(device)
        syn_torch = torch.from_numpy(syn_all).float().to(device)
        if gt_torch.shape[1] != 4 or syn_torch.shape[1] != 4:
            logger.warning(f"Skipping MMD for {class_name}, PCLs have != 4 features.")
            continue

        try:
            with torch.no_grad():
                final[key_xy] = mmd_loss_compute(gt_torch[:, :2], syn_torch[:, :2]).item()
                final[key_rcs] = mmd_loss_compute(gt_torch[:, 2:3], syn_torch[:, 2:3]).item()
                final[key_doppler] = mmd_loss_compute(gt_torch[:, 3:4], syn_torch[:, 3:4]).item()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM for {class_name}. Skipping global invariant MMD.")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise

    return final
