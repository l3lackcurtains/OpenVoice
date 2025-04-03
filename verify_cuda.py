import torch
import sys
import os

def check_cuda():
    print("\nPyTorch CUDA Information:")
    print("========================")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available for PyTorch")
        sys.exit(1)
        
    # CUDA Environment Variables
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')
    torch_cuda_arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', 'Not Set')
    
    print("\nCUDA Environment:")
    print("=================")
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    print(f"TORCH_CUDA_ARCH_LIST: {torch_cuda_arch_list}")
    
    # CUDA Details
    print("\nCUDA Details:")
    print("=============")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Device Properties
    device_props = torch.cuda.get_device_properties(0)
    print("\nDevice Properties:")
    print("=================")
    print(f"Device Name: {device_props.name}")
    print(f"Compute Capability: {device_props.major}.{device_props.minor}")
    print(f"Total Memory: {device_props.total_memory / 1e9:.2f} GB")
    print(f"Multi Processor Count: {device_props.multi_processor_count}")
    
    # Test CUDA Computation
    print("\nRunning CUDA Computation Test:")
    print("============================")
    try:
        # Matrix multiplication test
        size = 1000
        print(f"Testing {size}x{size} matrix multiplication...")
        x = torch.randn(size, size, device='cuda')
        y = torch.matmul(x, x)
        del x, y  # Clean up GPU memory
        print("Matrix multiplication test: Passed ✓")
        
        # Memory allocation test
        print("\nTesting GPU memory allocation...")
        memory_before = torch.cuda.memory_allocated()
        test_tensor = torch.ones(1000000, device='cuda')
        memory_after = torch.cuda.memory_allocated()
        memory_diff = memory_after - memory_before
        print(f"Memory allocation test: Passed ✓ (Allocated {memory_diff/1e6:.2f} MB)")
        del test_tensor  # Clean up GPU memory
        
        # CUDA synchronization test
        print("\nTesting CUDA synchronization...")
        torch.cuda.synchronize()
        print("CUDA synchronization test: Passed ✓")
        
        print("\nAll CUDA tests passed successfully! ✓")
        
    except Exception as e:
        print(f"\nError during CUDA test: {str(e)}")
        sys.exit(1)
    finally:
        # Clean up
        torch.cuda.empty_cache()

def set_cuda_environment():
    """Set CUDA environment variables if not already set"""
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TORCH_CUDA_ARCH_LIST' not in os.environ:
        os.environ['TORCH_CUDA_ARCH_LIST'] = '7.0;7.5;8.0;8.6'

if __name__ == "__main__":
    set_cuda_environment()
    check_cuda()
