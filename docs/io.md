# IO & Serialization

FastNN supports saving and loading models using safetensors format.

## Saving Models

```python
import fastnn as fnn

# Build and train model
model = fnn.models.MLP(input_dim=784, hidden_dims=[256, 128], output_dim=10)
# ... training code ...

# Save model
fnn.save_model(model, "model.safetensors")
```

## Loading Models

```python
import fastnn as fnn

# Create model architecture (must match saved architecture)
model = fnn.models.MLP(input_dim=784, hidden_dims=[256, 128], output_dim=10)

# Load weights
fnn.load_model(model, "model.safetensors")

# Use for inference
model.eval()
with fnn.no_grad():
    prediction = model(test_input)
```

## Complete Save/Load Example

```python
import fastnn as fnn

# === Training Phase ===
model = fnn.models.MLP(input_dim=10, hidden_dims=[32, 16], output_dim=2)
optimizer = fnn.Adam(model.parameters(), lr=1e-3)

# ... training loop ...

# Save trained model
fnn.save_model(model, "trained_model.safetensors")
print("Model saved!")

# === Inference Phase ===
# Load model
loaded_model = fnn.models.MLP(input_dim=10, hidden_dims=[32, 16], output_dim=2)
fnn.load_model(loaded_model, "trained_model.safetensors")

loaded_model.eval()
with fnn.no_grad():
    test_input = fnn.randn(5, 10)
    predictions = loaded_model(test_input)
    print(predictions.numpy())
```

## DLPack Interop

FastNN tensors support DLPack for interoperability with other libraries:

```python
import fastnn as fnn

x = fnn.randn(3, 4)

# Get DLPack capsule (for interoperability with other ML frameworks)
dlpack_cap = x.to_dlpack()

# Create tensor from DLPack
y = fnn.from_dlpack(dlpack_cap)
```

## Allocator Statistics

Monitor memory usage:

```python
# Get allocator statistics
stats = fnn.allocator_stats()
print(stats)
```

This returns information about:
- Total allocated memory
- Number of allocations
- Cache statistics

## Registered Operations

List all registered operations:

```python
ops = fnn.list_registered_ops()
print(ops)
```

This shows all tensor operations available in the library.
