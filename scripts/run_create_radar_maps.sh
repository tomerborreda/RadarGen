#!/bin/bash

# Generate radar maps using torchrun for multi-GPU support
#
# Usage:
#   bash scripts/run_create_radar_maps.sh <num_gpus> [additional_args]
#
# Examples:
#   bash scripts/run_create_radar_maps.sh 8 --config_path configs/preprocessing/truckscenes_radar_maps.yaml
#   bash scripts/run_create_radar_maps.sh 8 --config_path configs/preprocessing/nuscenes_radar_maps.yaml

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Add the parent directory to PYTHONPATH
export PYTHONPATH="${PARENT_DIR}:${PYTHONPATH}"

# Parse arguments
NUM_GPUS=${1:-8}
shift 1  # Remove first argument, pass remaining to python script

echo "Creating radar maps using ${NUM_GPUS} GPUs"

# Run the script with torchrun (single node, n GPUs)
torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" \
    "${SCRIPT_DIR}/create_radar_maps.py" "$@"
