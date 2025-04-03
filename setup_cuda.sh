#!/bin/bash

# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"

# Print current settings
echo "CUDA Configuration:"
echo "==================="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"

# Check NVIDIA driver and CUDA installation
if command -v nvidia-smi &> /dev/null; then
    echo -e "\nGPU Information:"
    echo "==================="
    nvidia-smi
else
    echo -e "\nWarning: nvidia-smi not found. Please check NVIDIA driver installation."
fi

# Check CUDA installation
if [ -d "/usr/local/cuda" ]; then
    echo -e "\nCUDA Installation:"
    echo "==================="
    echo "CUDA Path: $(ls -l /usr/local/cuda)"
    echo "CUDA Version: $(/usr/local/cuda/bin/nvcc --version | grep "release" | awk '{print $5}')"
else
    echo -e "\nWarning: CUDA installation not found in /usr/local/cuda"
fi

# Run the verification script
echo -e "\nRunning PyTorch CUDA verification..."
python verify_cuda.py

echo -e "\nSetup complete! Environment variables have been set for this session."
echo "To make these settings permanent, add them to your ~/.bashrc file."
