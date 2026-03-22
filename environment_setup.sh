#!/usr/bin/env bash
set -e

CONDA_ENV=${1:-"radargen"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "RadarGen Environment Setup"
echo "============================================"
echo "Environment: $CONDA_ENV"
echo "============================================"

# Initialize git submodules if needed
if [ -f "$SCRIPT_DIR/.gitmodules" ]; then
    echo "Initializing git submodules..."
    cd "$SCRIPT_DIR"
    git submodule update --init --recursive
fi

# Create and activate conda environment
if [ -n "$CONDA_ENV" ]; then
    eval "$(conda shell.bash hook)"
    
    echo "Creating conda environment: $CONDA_ENV"
    conda create -n $CONDA_ENV python=3.11.0 -y
    conda activate $CONDA_ENV
else
    echo "Skipping conda environment creation. Make sure you have the correct environment activated."
fi

# Ensure setuptools is available (provides pkg_resources, needed to build wheels)
python3 -m pip install --upgrade setuptools
python3 -m pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support
echo "Installing PyTorch with CUDA 12.1..."
python3 -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install RadarGen and dependencies
echo "Installing RadarGen..."
python3 -m pip install -e "$SCRIPT_DIR"

# Install mmcv 1.x without build isolation (no Python 3.11 wheel available)
python3 -m pip install mmcv==1.7.2 --no-build-isolation

# Upgrade timm (required for compatibility)
python3 -m pip install --upgrade timm

echo "============================================"
echo "Core installation complete!"
echo "============================================"

# Install foundation models (UniDepth and UFM)
echo ""
echo "============================================"
echo "Installing Foundation Models (UniDepth, UFM)"
echo "============================================"

# Install CUDA toolkit for compiling ops
echo "Installing CUDA toolkit..."
conda install -c nvidia/label/cuda-12.1.1 cuda-toolkit cuda-nvcc -y

# Install UniDepth
echo ""
echo "Installing UniDepth..."
UNIDEPTH_DIR="$SCRIPT_DIR/UniDepth"
if [ ! -d "$UNIDEPTH_DIR" ] || [ -z "$(ls -A "$UNIDEPTH_DIR" 2>/dev/null)" ]; then
    echo "Error: UniDepth submodule not found. Please run:"
    echo "  git submodule update --init --recursive"
    exit 1
fi
cd "$UNIDEPTH_DIR"
python3 -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cu121

# Compile UniDepth CUDA ops
echo "Compiling UniDepth CUDA ops..."
if [ -d "./unidepth/ops/extract_patches" ]; then
    cd ./unidepth/ops/extract_patches
    bash compile.sh
    cd "$SCRIPT_DIR"
else
    echo "Warning: UniDepth ops directory not found, skipping compilation"
    cd "$SCRIPT_DIR"
fi

# Install UFM
echo ""
echo "Installing UFM..."
UFM_DIR="$SCRIPT_DIR/UFM"
if [ ! -d "$UFM_DIR" ] || [ -z "$(ls -A "$UFM_DIR" 2>/dev/null)" ]; then
    echo "Error: UFM submodule not found. Please run:"
    echo "  git submodule update --init --recursive"
    exit 1
fi

# Install UniCeption (UFM dependency)
echo "Installing UniCeption..."
cd "$UFM_DIR/UniCeption"
python3 -m pip install -e .

# Install UFM
cd "$UFM_DIR"
python3 -m pip install -e .
cd "$SCRIPT_DIR"

echo ""
echo "============================================"
echo "Installation complete!"
echo "============================================"
