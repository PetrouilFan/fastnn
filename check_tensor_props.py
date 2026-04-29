#!/usr/bin/env python3
"""Check FastNN tensor properties for matmul fast path verification."""

import fastnn

def main():
    # Create two 1024x1024 tensors with normal distribution
    a = fastnn.randn([1024, 1024])
    b = fastnn.randn([1024, 1024])
    
    # Check requires_grad status (should be False by default)
    print(f"Tensor A requires_grad: {a.requires_grad}")
    print(f"Tensor B requires_grad: {b.requires_grad}")
    
    # Print properties for both tensors
    for i, tensor in enumerate([a, b], 1):
        print(f"\nTensor {i}:")
        print(f"  ndim: {tensor.ndim}")
        print(f"  shape: {tensor.shape}")
        print(f"  strides: {tensor.debug_strides()}")
        print(f"  is_contiguous: {tensor.is_contiguous}")
        print(f"  dtype: {tensor.dtype}")
        print(f"  device: {tensor.device}")
    
    # Fast path conditions for matmul:
    # - Both tensors should be contiguous
    # - Both should have requires_grad=False (no autograd)
    # - Typically 2D matrices with shape [M, K] and [K, N]
    print("\nFast path conditions check:")
    print(f"  A contiguous: {a.is_contiguous}")
    print(f"  B contiguous: {b.is_contiguous}")
    print(f"  A requires_grad: {a.requires_grad}")
    print(f"  B requires_grad: {b.requires_grad}")
    
    both_contiguous = a.is_contiguous and b.is_contiguous
    no_grad = not a.requires_grad and not b.requires_grad
    print(f"\n  Meets fast path (contiguous + no grad): {both_contiguous and no_grad}")

if __name__ == "__main__":
    main()
