#!/bin/bash
# ============================================================================
# Setup script for Emotive-Spider PPO Training Environment
# ============================================================================
# This script creates a UV virtual environment and installs all required
# dependencies for training a spider robot to walk using PPO with MuJoCo MJX.
# ============================================================================

set -e  # Exit on any error

echo "============================================"
echo "  Emotive-Spider Training Environment Setup"
echo "============================================"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' is not installed. Install it with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create virtual environment
echo ""
echo "[1/3] Creating UV virtual environment..."
uv venv .venv --python 3.11

# Activate virtual environment
echo ""
echo "[2/3] Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo ""
echo "[3/3] Installing dependencies..."
uv pip install \
    mujoco \
    mujoco-mjx \
    jax[cuda12] \
    flax \
    optax \
    warp-lang \
    numpy \
    matplotlib \
    tqdm

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "Then start training with:"
echo "  python train.py"
echo ""
