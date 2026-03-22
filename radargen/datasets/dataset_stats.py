#!/usr/bin/env python3
"""Compute radar point cloud statistics over a subset of dataset keyframes.

Usage:
    python radargen/datasets/dataset_stats.py \
        --dataset_name truckscenes \
        --dataset_dir /datasets/RadarGen/man-truckscenes \
        --dataset_version v1.0-trainval \
        --split train \
        --max_samples 2000

    
    python -m radargen.datasets.dataset_stats --dataset_name truckscenes --dataset_dir /path/to/truckscenes --dataset_version v1.0-trainval --split train --max_samples 20000
    python -m radargen.datasets.dataset_stats --dataset_name nuscenes --dataset_dir /path/to/nuscenes --dataset_version v1.0-trainval --split train --max_samples 20000
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyrallis
from tqdm.auto import tqdm

from radargen.datasets.registry import get_adapter


@dataclass
class DatasetStatsConfig:
    dataset_name: str
    dataset_dir: Optional[str] = None
    dataset_version: str = "v1.0-trainval"
    split: str = "train"
    max_samples: int = 2000
    nsweeps: int = 1
    min_distance: float = 0.0


def _field_stats(values: np.ndarray) -> dict:
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p5": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _print_stats(label: str, stats: dict) -> None:
    print(
        f"{label:<18s}"
        f"{stats['count']:>9d}"
        f"{stats['mean']:>11.2f}"
        f"{stats['std']:>10.2f}"
        f"{stats['min']:>10.2f}"
        f"{stats['p5']:>10.2f}"
        f"{stats['p25']:>10.2f}"
        f"{stats['p50']:>10.2f}"
        f"{stats['p75']:>10.2f}"
        f"{stats['p95']:>10.2f}"
        f"{stats['max']:>10.2f}"
    )


def main() -> None:
    cfg = pyrallis.parse(config_class=DatasetStatsConfig)

    print(f"Initializing {cfg.dataset_name} adapter...")
    adapter = get_adapter(
        name=cfg.dataset_name,
        dataset_dir=cfg.dataset_dir,
        dataset_version=cfg.dataset_version,
    )

    n_points_list = []
    x_list, y_list, z_list = [], [], []
    rcs_list, doppler_list = [], []
    samples_collected = 0

    pbar = tqdm(desc="Collecting samples", total=cfg.max_samples, unit="sample")
    for _scene_token, keyframes in adapter.iter_keyframes(cfg.split):
        if samples_collected >= cfg.max_samples:
            break
        for sample in keyframes:
            if samples_collected >= cfg.max_samples:
                break
            sample_data = sample['data'] if isinstance(sample, dict) and 'data' in sample else sample
            pcl = adapter.load_radar_pointcloud(
                sample_data,
                nsweeps=cfg.nsweeps,
                min_distance=cfg.min_distance,
            )
            if len(pcl) == 0:
                raise ValueError(f"No points found for sample {sample_data}")
            n_points_list.append(len(pcl))
            x_list.append(pcl[:, 0])
            y_list.append(pcl[:, 1])
            z_list.append(pcl[:, 2])
            rcs_list.append(pcl[:, 6])
            doppler_list.append(pcl[:, 7])
            samples_collected += 1
            pbar.update(1)
    pbar.close()

    sep = "=" * 120
    col_header = (
        f"{'Field':<18s}"
        f"{'count':>9s}"
        f"{'mean':>11s}"
        f"{'std':>10s}"
        f"{'min':>10s}"
        f"{'p5':>10s}"
        f"{'p25':>10s}"
        f"{'p50':>10s}"
        f"{'p75':>10s}"
        f"{'p95':>10s}"
        f"{'max':>10s}"
    )

    print()
    print(f"Dataset Statistics: {cfg.dataset_name} ({cfg.split}, {samples_collected} samples)")
    print(sep)
    print(col_header)
    print("-" * 120)

    n_pts = np.array(n_points_list, dtype=np.float64)
    all_x = np.concatenate(x_list)
    all_y = np.concatenate(y_list)
    all_z = np.concatenate(z_list)
    all_rcs = np.concatenate(rcs_list)
    all_doppler = np.concatenate(doppler_list)

    _print_stats("Points/sample", _field_stats(n_pts))
    _print_stats("x (m)", _field_stats(all_x))
    _print_stats("y (m)", _field_stats(all_y))
    _print_stats("z (m)", _field_stats(all_z))
    _print_stats("RCS (dBsm)", _field_stats(all_rcs))
    _print_stats("Doppler (m/s)", _field_stats(all_doppler))

    print(sep)
    print(f"Total points across all samples: {int(np.sum(n_pts)):,}")


if __name__ == "__main__":
    main()
