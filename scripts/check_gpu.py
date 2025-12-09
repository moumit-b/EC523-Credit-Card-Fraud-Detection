"""
Quick script to check if GPU is available and being used.
Run this before running the main experiments.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from config.config import DEVICE

print("=" * 80)
print("GPU AVAILABILITY CHECK")
print("=" * 80)

print(f"\nPyTorch version: {torch.__version__}")

# Check CUDA availability
print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

# Check selected device
print(f"\nSelected device from config: {DEVICE}")

# Test tensor creation on device
try:
    test_tensor = torch.randn(100, 100).to(DEVICE)
    print(f"✅ Successfully created test tensor on {DEVICE}")
    print(f"   Tensor device: {test_tensor.device}")

    # Test computation
    result = torch.matmul(test_tensor, test_tensor)
    print(f"✅ Successfully performed computation on {DEVICE}")

    if DEVICE.type == 'cuda':
        print(f"\n🎉 GPU IS READY! Your training will use: {torch.cuda.get_device_name(0)}")
        print(f"   Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"\n⚡ Expected speedup: 10-50x faster than CPU")
    elif DEVICE.type == 'mps':
        print(f"\n🎉 Apple Metal GPU IS READY!")
        print(f"\n⚡ Expected speedup: 3-10x faster than CPU")
    else:
        print(f"\n⚠️  Running on CPU. Training will be slower (~20-30 min per model)")
        print(f"   Consider using a GPU for faster training.")

except Exception as e:
    print(f"❌ Error testing device: {e}")
    print(f"   Will fall back to CPU")

print("\n" + "=" * 80)
