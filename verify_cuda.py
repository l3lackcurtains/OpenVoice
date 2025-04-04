import torch
import sys
import os

def check_cuda():
    print("\nPyTorch CUDA Information:")
    print("========================")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        # During build, we only verify PyTorch installation
        if os.environ.get('DOCKER_BUILD') == '1':
            print("Building container - skipping CUDA verification")
            sys.exit(0)
        else:
            print("Warning: CUDA is not available for PyTorch")
            sys.exit(1)
        
    # CUDA Environment Variables
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')
    torch_cuda_arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', 'Not Set')
    
    print("\nCUDA Environment:")
    print("=================")
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    print(f"TORCH_CUDA_ARCH_LIST: {torch_cuda_arch_list}")
    
    # Only attempt device queries when not building
    if os.environ.get('DOCKER_BUILD') != '1':
        # CUDA Details
        print("\nCUDA Details:")
        print("=============")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        # Test CUDA Computation
        print("\nRunning CUDA Computation Test:")
        print("============================")
        try:
            size = 1000
            x = torch.randn(size, size, device='cuda')
            y = torch.matmul(x, x)
            del x, y
            print("CUDA computation test passed ✓")
        except Exception as e:
            print(f"CUDA computation test failed: {e}")
            sys.exit(1)

def set_cuda_environment():
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TORCH_CUDA_ARCH_LIST' not in os.environ:
        os.environ['TORCH_CUDA_ARCH_LIST'] = '7.0;7.5;8.0;8.6'

if __name__ == "__main__":
    set_cuda_environment()
    check_cuda()
