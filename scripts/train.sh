#/bin/bash
set -e

# If you don't install with 'pip install .' then you need to set the PYTHONPATH manually.
# Get the parent directory of where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
# Set PYTHONPATH relative to the script's parent directory
export PYTHONPATH="${PARENT_DIR}:$PYTHONPATH"

export OMP_NUM_THREADS=8

export TORCH_DISTRIBUTED_DEBUG=INFO

WORK_DIR=${WORK_DIR:-"output/debug"} # Default is output/debug, set WORK_DIR if the work directory is different
NUM_GPUS=${NUM_GPUS:-8} # Default is 8 GPUs, set NUM_GPUS if the number of GPUs is different

config=$1
shift

TRITON_PRINT_AUTOTUNING=1 \
    torchrun --nproc_per_node=$NUM_GPUS --master_port=15432 \
        scripts/train.py \
        --config_path=$config \
        --work_dir=$WORK_DIR \
        --name=train_radargen_600m_512px \
        --tracker_project_name=train_radargen_600m_512px \
        --report_to=wandb \
        "$@"
