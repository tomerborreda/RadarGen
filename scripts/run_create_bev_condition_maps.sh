#!/bin/bash
# Generate BEV conditioning maps using multi-GPU distributed training
#
# Usage:
#   bash scripts/run_create_bev_condition_maps.sh <num_gpus> [additional_args]
#
# Examples:
#   bash scripts/run_create_bev_condition_maps.sh 8 --config_path configs/preprocessing/truckscenes_bev_condition_maps.yaml
#   bash scripts/run_create_bev_condition_maps.sh 8 --config_path configs/preprocessing/nuscenes_bev_condition_maps.yaml

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Add the parent directory to PYTHONPATH
export PYTHONPATH="${PARENT_DIR}:${PYTHONPATH}"

# Parse arguments
NUM_GPUS=${1:-8}
shift 1  # Remove first argument, pass remaining to python script

echo "Creating BEV condition maps using ${NUM_GPUS} GPUs"

# Run generation with torchrun
torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" \
    "${SCRIPT_DIR}/create_bev_condition_maps.py" "$@"
