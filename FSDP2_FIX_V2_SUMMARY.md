# Liger Kernel FSDP2 Compatibility Fix V2

## Problem with V1
The initial fix created full-size zero tensors and placed local gradients in them, which caused PyTorch stream synchronization errors:
```
RuntimeError: opt_ready_stream && opt_parent_stream INTERNAL ASSERT FAILED
```

## Solution V2: Use DTensor Native APIs

The revised approach uses DTensor's native `from_local` API to properly wrap local gradients back into DTensor format, avoiding manual tensor construction and stream synchronization issues.

## Key Changes

### 1. Import DTensor Safely
```python
try:
    from torch.distributed._tensor import DTensor
except ImportError:
    DTensor = None
```

### 2. Track DTensor State in Forward
```python
# Check if inputs are DTensors
weight_is_dtensor = DTensor is not None and isinstance(weight, DTensor)

# Extract local tensors but keep DTensor references
if weight_is_dtensor:
    weight_local = weight._local_tensor
    original_weight_dtensor = weight  # Keep for backward
else:
    weight_local = weight
    original_weight_dtensor = None
```

### 3. Return DTensor References from Forward
```python
# Return additional information for backward pass
return loss, z_loss, token_accuracy, grad_input, grad_weight, grad_bias, 
       weight_is_dtensor, bias_is_dtensor, original_weight_dtensor, original_bias_dtensor
```

### 4. Reconstruct DTensor Gradients in Backward
```python
# Use DTensor.from_local to wrap gradients properly
if ctx.weight_is_dtensor and grad_weight is not None and original_weight_dtensor is not None:
    grad_weight = DTensor.from_local(
        grad_weight,
        original_weight_dtensor.device_mesh,
        original_weight_dtensor.placements,
        run_check=False  # Skip checks for performance
    )
```

## Why This Works Better

1. **No Full Tensor Creation**: We don't create full-size tensors, avoiding memory overhead
2. **Native DTensor APIs**: Uses PyTorch's official DTensor reconstruction method
3. **Preserves Stream Order**: Doesn't interfere with CUDA stream synchronization
4. **Maintains Placement**: Gradients have the same sharding as the original tensors

## Benefits

- ✅ Fixes the gradient shape mismatch error
- ✅ Avoids PyTorch internal stream errors
- ✅ Memory efficient (only local tensors in memory)
- ✅ Compatible with FSDP2's gradient accumulation
- ✅ Works seamlessly with regular tensors

## Testing

The fix ensures:
1. Local gradients are computed on sharded data
2. Gradients are properly wrapped back as DTensors
3. FSDP2 can handle the gradient accumulation across ranks
4. No interference with PyTorch's internal stream management