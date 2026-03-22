#!/usr/bin/env python3
"""Generate ground truth radar maps for training RadarGen.

This script generates Point Density, RCS, and Doppler maps from radar point clouds.
Supports distributed generation across multiple GPUs via torchrun.

Usage:
    # Single GPU
    python scripts/create_radar_maps.py --config path/to/config.yaml

    # Multi-GPU (via torchrun)
    torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/create_radar_maps.py --config path/to/config.yaml

Example config.yaml:
    dataset_name: truckscenes
    dataset_dir: /path/to/dataset
    save_dir: /path/to/output
    image_size: 512
    point_limit: 50.0
    nsweeps: 1
    sigma: 2.0
    dataset_version: v1.0-trainval
    filter_for_split: train
"""

import os
import sys
import time
from datetime import timedelta

import pyrallis
import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from radargen.radar_maps import (
    RadarMapCreationConfig,
    create_radar_maps_for_scene,
)
from radargen.datasets.registry import get_adapter
from radargen.datasets.base import distribute_across_ranks


def main():
    try:
        cfg = pyrallis.parse(config_class=RadarMapCreationConfig)
    except SystemExit:
        print("\nERROR: Missing configuration. Please provide a config file:")
        print("  Single GPU:")
        print("    python scripts/create_radar_maps.py --config_path configs/preprocessing/truckscenes_radar_maps.yaml")
        print("  Multi-GPU (8 GPUs):")
        print("    bash scripts/run_create_radar_maps.sh 8 --config_path configs/preprocessing/truckscenes_radar_maps.yaml\n")
        sys.exit(1)

    # --- Distributed init (single-node, multi-GPU via torchrun) ---
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    using_dist = world_size > 1 and torch.cuda.is_available()
    if using_dist:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(hours=24))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank, world_size, local_rank = 0, 1, 0

    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")

    print(f'Starting: Rank {rank}, World size {world_size}, Local rank {local_rank}')

    # Extract config parameters
    image_size = cfg.image_size
    point_limit = cfg.point_limit
    filter_camera_views = cfg.filter_camera_views
    nsweeps = cfg.nsweeps
    sigma = cfg.sigma

    # Check time
    start_time = time.time()

    assert os.path.isdir(cfg.dataset_dir), f'Dataset directory {cfg.dataset_dir} does not exist'

    # Create save directory if it doesn't exist
    os.makedirs(cfg.save_dir, exist_ok=True)
    print(f'Saving radar maps to: {cfg.save_dir}')

    # Initialize adapter for the specified dataset
    print(f'Initializing {cfg.dataset_name} dataset adapter...')
    adapter = get_adapter(
        name=cfg.dataset_name,
        dataset_dir=str(cfg.dataset_dir),
        dataset_version=cfg.dataset_version,
    )

    # Get all scenes using iterator pattern
    if cfg.filter_for_split is not None:
        print(f'Filtering for split: {cfg.filter_for_split}')
        scenes = list(adapter.iter_scenes(split=cfg.filter_for_split))
    else:
        # Get all scenes from all available splits
        scenes = []
        for split in ['train', 'val']:
            scenes.extend(list(adapter.iter_scenes(split=split)))

    # Distribute scenes across GPUs
    local_scenes = distribute_across_ranks(scenes, rank, world_size)
    print(f'Rank {rank}: Processing {len(local_scenes)} scenes (out of {len(scenes)} total)')

    # Single progress bar per rank (works reliably over SSH)
    # Use postfix to show sample-level progress
    scene_iterator = tqdm(
        local_scenes,
        desc=f"[Rank {rank}] Scenes",
        position=rank,
        leave=True,
        file=sys.stdout,
        dynamic_ncols=True,
        miniters=1
    )

    for scene_token, samples in scene_iterator:
        # Define callback to update progress bar postfix with sample info
        def update_sample_progress(sample_idx, total_samples):
            scene_iterator.set_postfix_str(
                f"Scene: {scene_token} | Sample: {sample_idx+1}/{total_samples}"
            )

        create_radar_maps_for_scene(
            adapter=adapter,
            scene_token=scene_token,
            samples=samples,
            save_dir_path=str(cfg.save_dir),
            device=device,
            image_size=image_size,
            point_limit=point_limit,
            filter_camera_views=filter_camera_views,
            nsweeps=nsweeps,
            sigma_range=(sigma, sigma),
            progress_callback=update_sample_progress,
        )

    end_time = time.time()
    if using_dist:
        dist.barrier()
        dist.destroy_process_group()
    print(f'Done! Rank {rank}. Time taken: {(end_time - start_time) / 60:.2f} minutes')


if __name__ == "__main__":
    main()
