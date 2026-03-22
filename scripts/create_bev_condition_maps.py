#!/usr/bin/env python3
"""Create BEV conditioning maps for multi-view camera data.

This script creates BEV color, segmentation, and velocity maps from multi-view camera images.
Supports distributed creation across multiple GPUs via torchrun.

Usage:
    # Single GPU
    python scripts/create_bev_condition_maps.py --config_path path/to/config.yaml

    # Multi-GPU (via torchrun)
    torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/create_bev_condition_maps.py --config_path path/to/config.yaml

Example config.yaml:
    dataset_name: truckscenes
    dataset_dir: /path/to/dataset
    save_dir: /path/to/output
    resolution: 512
    dataset_version: v1.0-trainval
    split: train
"""

import os
import sys
import time
from datetime import timedelta

import pyrallis
import torch
import torch.distributed as dist
from tqdm import tqdm

from radargen.bev_condition_maps import (
    BEVConditionConfig,
    create_bev_maps_for_scene,
)
from radargen.datasets.registry import get_adapter
from radargen.datasets.base import distribute_across_ranks
from radargen.bev_condition_maps.foundation_models import load_models


def main():
    try:
        cfg = pyrallis.parse(config_class=BEVConditionConfig)
    except SystemExit:
        print("\nERROR: Missing configuration. Please provide a config file:")
        print("  Single GPU:")
        print("    python scripts/create_bev_condition_maps.py --config_path configs/preprocessing/truckscenes_bev_maps.yaml")
        print("  Multi-GPU (8 GPUs):")
        print("    bash scripts/run_create_bev_condition_maps.sh 8 --config_path configs/preprocessing/truckscenes_bev_maps.yaml\n")
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

    if rank == 0:
        print(f"Starting BEV map creation with {world_size} process(es)")
        print(f"Dataset: {cfg.dataset_name}")
        print(f"Dataset dir: {cfg.dataset_dir}")
        print(f"Save dir: {cfg.save_dir}")
        print(f"Resolution: {cfg.resolution}")

    # Check time
    start_time = time.time()

    assert os.path.isdir(cfg.dataset_dir), f'Dataset directory {cfg.dataset_dir} does not exist'

    # Create save directory if it doesn't exist
    os.makedirs(cfg.save_dir, exist_ok=True)
    if rank == 0:
        print(f'Saving BEV maps to: {cfg.save_dir}')

    # Initialize adapter for the specified dataset
    if rank == 0:
        print(f'Initializing {cfg.dataset_name} dataset adapter...')
    adapter = get_adapter(
        name=cfg.dataset_name,
        dataset_dir=str(cfg.dataset_dir),
        dataset_version=cfg.dataset_version,
        camera_views=cfg.camera_views,
    )

    if rank == 0:
        print(f"Coordinate range: {adapter.normalization.coordinate_range}m")
        print(f"Camera frequency: {adapter.normalization.camera_freq}Hz")
        print(f"Camera views: {adapter.camera_views}")

    # Load foundation models
    if rank == 0:
        print("Loading foundation models...")
    depth, seg, seg_proc, flow = load_models(device)
    models = {
        'depth': depth,
        'segmentation': seg,
        'segmentation_processor': seg_proc,
        'flow': flow,
    }

    # Apply torch.compile if enabled
    if cfg.use_torch_compile:
        if rank == 0:
            print("Applying torch.compile() to foundation models...")
        depth = torch.compile(depth, mode="reduce-overhead")
        seg = torch.compile(seg, mode="reduce-overhead")
        flow = torch.compile(flow, mode="reduce-overhead")
        models['depth'] = depth
        models['segmentation'] = seg
        models['flow'] = flow
        if rank == 0:
            print("torch.compile() enabled for foundation models")

    # Print optimization settings
    if rank == 0:
        print(f"\nOptimization settings:")
        print(f"  Batched inference: {cfg.use_batched_inference}")
        print(f"  Async I/O: {cfg.use_async_io}")
        print(f"  torch.compile: {cfg.use_torch_compile}")
        print(f"  Mixed precision: {cfg.use_mixed_precision}")

    # Get all scenes using iterator pattern
    if cfg.split is not None:
        if rank == 0:
            print(f'Filtering for split: {cfg.split}')
        scenes = list(adapter.iter_scenes(split=cfg.split))
    else:
        # Get all scenes from all available splits
        scenes = []
        for split in ['train', 'val']:
            scenes.extend(list(adapter.iter_scenes(split=split)))

    # Distribute scenes across GPUs
    local_scenes = distribute_across_ranks(scenes, rank, world_size)
    if rank == 0:
        print(f"Total scenes: {len(scenes)}")
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

        create_bev_maps_for_scene(
            adapter=adapter,
            scene_token=scene_token,
            samples=samples,
            models=models,
            save_dir=cfg.save_dir,
            device=device,
            resolution=cfg.resolution,
            use_batched_inference=cfg.use_batched_inference,
            use_async_io=cfg.use_async_io,
            progress_callback=update_sample_progress,
        )

    end_time = time.time()
    if using_dist:
        dist.barrier()
        if rank == 0:
            print("All processes finished. Cleaning up...")
        dist.destroy_process_group()

    if rank == 0:
        print(f'Done! BEV map creation complete!')
        print(f'Time taken: {(end_time - start_time) / 60:.2f} minutes')


if __name__ == "__main__":
    main()
