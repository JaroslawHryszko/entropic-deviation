#!/bin/bash
# install_gpu.sh - Script to install llama-cpp-python with multi-GPU support

# Check if CUDA is installed
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: NVIDIA CUDA toolkit not found!"
    echo "Please install CUDA toolkit first: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

# Install regular dependencies
pip install -r requirements.txt

# Detect GPU architecture
echo "Detecting GPU architecture..."
GPU_ARCH=$(nvcc --version | grep "cuda_" | grep -o "sm_[0-9]*" | sed 's/sm_//')

if [ -z "$GPU_ARCH" ]; then
    echo "Could not detect GPU architecture automatically."
    echo "Using default architecture setting: all"
    ARCH_SETTING="all"
else
    echo "Detected GPU architecture: $GPU_ARCH"
    ARCH_SETTING=$GPU_ARCH
fi

echo "Installing llama-cpp-python with multi-GPU support..."

# Set environment variables and install
export CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$ARCH_SETTING"
export FORCE_CMAKE=1
pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python==0.2.72

echo "Installation complete!"
echo "To verify installation, run: python test_gpu.py"