#!/bin/bash
# GEMS Environment Setup
# ======================
# One-click setup: conda env + PyTorch + dependencies + editable install.
# Requires: conda (Miniconda or Anaconda)
# GPU: auto-detected by PyTorch. Adjust CUDA version below if needed.

set -e

command -v conda >/dev/null 2>&1 || { echo "Error: conda not found. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html"; exit 1; }

echo ""
echo "=== GEMS Environment Setup ==="
echo ""

echo "[1/5] Creating conda env 'gems' (Python 3.10)..."
conda create -n gems python=3.10 -y

echo ""
echo "[2/5] Installing PyTorch 2.5.1 (CUDA 12.1)..."
echo "       (If your CUDA version differs, change pytorch-cuda=12.1 below)"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gems
conda install pytorch==2.5.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

echo ""
echo "[3/5] Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "[4/5] Ensuring transformers==5.9.0 (Qwen3.5 requirement)..."
pip install "transformers==5.9.0" --upgrade

echo ""
echo "[5/5] Installing GEMS package (editable mode)..."
pip install -e .

echo ""
echo "=== Done ==="
echo "Activate with:  conda activate gems"
echo "Then run:       python demo.py"
echo ""
