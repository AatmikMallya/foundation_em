#!/usr/bin/env python3

import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    
    # Test torch.compile
    print("Testing torch.compile...")
    @torch.compile
    def simple_func(x):
        return x * 2
    
    x = torch.tensor([1.0])
    result = simple_func(x)
    print(f"torch.compile test passed: {result}")
    
    print("✅ Environment test PASSED")
    
except Exception as e:
    print(f"❌ Environment test FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 