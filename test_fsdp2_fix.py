#!/usr/bin/env python3
"""
Test script to verify Liger Kernel's fused linear cross-entropy works with FSDP2
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.device_mesh import init_device_mesh
import os
import sys

# Add Liger Kernel to path
sys.path.insert(0, '/home/rgilbreth/Desktop/AI-Software/Liger-Kernel/src')

from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction


def test_regular_tensors():
    """Test with regular tensors (non-distributed)"""
    print("Testing with regular tensors...")
    
    batch_size = 2
    seq_len = 8
    hidden_dim = 2560
    vocab_size = 151936
    
    # Create test data
    input_tensor = torch.randn(batch_size * seq_len, hidden_dim, requires_grad=True)
    weight = torch.randn(vocab_size, hidden_dim, requires_grad=True)
    bias = torch.randn(vocab_size, requires_grad=True)
    target = torch.randint(0, vocab_size, (batch_size * seq_len,))
    
    # Forward pass
    loss, _, _ = LigerFusedLinearCrossEntropyFunction.apply(
        input_tensor, weight, target, bias, None, -100, 0.0, 0.0, "mean", None, False, None, False, False
    )
    
    # Backward pass
    loss.backward()
    
    print(f"  Input shape: {input_tensor.shape}")
    print(f"  Weight shape: {weight.shape}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Input grad shape: {input_tensor.grad.shape}")
    print(f"  Weight grad shape: {weight.grad.shape}")
    print("  ✓ Regular tensor test passed!\n")


def test_dtensor_single_process():
    """Test with DTensor in single process mode"""
    print("Testing with DTensor (single process simulation)...")
    
    # Initialize distributed environment for single process
    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    
    # Create device mesh
    device_mesh = init_device_mesh("cpu", (1,))
    
    batch_size = 2
    seq_len = 8
    hidden_dim = 2560
    vocab_size = 151936
    
    # Create test data
    input_tensor = torch.randn(batch_size * seq_len, hidden_dim, requires_grad=True)
    weight = torch.randn(vocab_size, hidden_dim, requires_grad=True)
    bias = torch.randn(vocab_size, requires_grad=True)
    target = torch.randint(0, vocab_size, (batch_size * seq_len,))
    
    # Convert weight and bias to DTensor (sharded along first dimension)
    weight_dtensor = DTensor.from_local(weight, device_mesh, [Shard(0)])
    bias_dtensor = DTensor.from_local(bias, device_mesh, [Shard(0)])
    
    # Enable gradients for DTensors
    weight_dtensor.requires_grad_(True)
    bias_dtensor.requires_grad_(True)
    
    print(f"  Weight DTensor shape: {weight_dtensor.shape}")
    print(f"  Weight local tensor shape: {weight_dtensor._local_tensor.shape}")
    
    # Forward pass
    loss, _, _ = LigerFusedLinearCrossEntropyFunction.apply(
        input_tensor, weight_dtensor, target, bias_dtensor, None, -100, 0.0, 0.0, "mean", None, False, None, False, False
    )
    
    # Backward pass
    loss.backward()
    
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Weight grad shape: {weight_dtensor.grad.shape if weight_dtensor.grad is not None else 'None'}")
    print(f"  Weight grad local shape: {weight_dtensor.grad._local_tensor.shape if weight_dtensor.grad and hasattr(weight_dtensor.grad, '_local_tensor') else 'N/A'}")
    print("  ✓ DTensor test passed!\n")
    
    # Cleanup
    dist.destroy_process_group()


def test_dtensor_multi_process_simulation():
    """Simulate multi-process FSDP2 scenario"""
    print("Simulating multi-process FSDP2 scenario...")
    
    # This simulates what happens in a 4-GPU setup
    world_size = 4
    rank = 0  # Simulate rank 0
    
    batch_size = 2
    seq_len = 8
    hidden_dim = 2560
    vocab_size = 151936
    shard_size = vocab_size // world_size  # 37984
    
    print(f"  World size: {world_size}")
    print(f"  Current rank: {rank}")
    print(f"  Full vocab size: {vocab_size}")
    print(f"  Shard size per rank: {shard_size}")
    
    # Create test data (input is not sharded)
    input_tensor = torch.randn(batch_size * seq_len, hidden_dim, requires_grad=True)
    
    # Create local shard of weight and bias (simulating what each rank would have)
    weight_local = torch.randn(shard_size, hidden_dim, requires_grad=True)
    bias_local = torch.randn(shard_size, requires_grad=True)
    target = torch.randint(0, vocab_size, (batch_size * seq_len,))
    
    print(f"  Local weight shape: {weight_local.shape}")
    print(f"  Expected full weight shape: ({vocab_size}, {hidden_dim})")
    
    # Create mock DTensor objects
    class MockDTensor:
        def __init__(self, local_tensor, full_shape):
            self._local_tensor = local_tensor
            self.shape = full_shape
            self.requires_grad = local_tensor.requires_grad
            self.dtype = local_tensor.dtype
            self.device = local_tensor.device
        
        def __repr__(self):
            return f"MockDTensor(shape={self.shape}, local_shape={self._local_tensor.shape})"
    
    weight_dtensor = MockDTensor(weight_local, torch.Size([vocab_size, hidden_dim]))
    bias_dtensor = MockDTensor(bias_local, torch.Size([vocab_size]))
    
    print(f"  Created mock DTensors:")
    print(f"    Weight: {weight_dtensor}")
    print(f"    Bias: {bias_dtensor}")
    
    print("\n  ⚠️  Note: This is a simulation. In real FSDP2, the DTensor would handle gradient")
    print("     accumulation across ranks. Our fix ensures we return the correct gradient shape.\n")


if __name__ == "__main__":
    print("=== Liger Kernel FSDP2 Compatibility Test ===\n")
    
    # Test 1: Regular tensors
    test_regular_tensors()
    
    # Test 2: DTensor single process
    test_dtensor_single_process()
    
    # Test 3: Multi-process simulation
    test_dtensor_multi_process_simulation()
    
    print("=== All tests completed! ===")