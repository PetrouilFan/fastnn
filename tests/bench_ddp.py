"""Benchmark script for Distributed Data Parallel (DDP) in fastnn.

Tests 1-GPU vs 2-GPU performance with bucketed AllReduce gradient synchronization.
"""

import fastnn as fnn
from fastnn.parallel import DataParallel
import time


def move_model_to_gpu(model, device_id):
    """Helper to move model parameters to specific GPU.

    This creates a new model with parameters on the target GPU.
    """
    # Get the model's layers and rebuild with GPU parameters
    new_layers = []
    for layer in model.model.layers:
        if hasattr(layer, "parameters"):
            params = layer.parameters()
            if len(params) == 2:  # Linear layer with weight + bias
                weight, bias = params[0], params[1]
                # Create new Linear layer with GPU tensors
                in_features = weight.shape[1]  # weight is [out, in]
                out_features = weight.shape[0]
                new_layer = fnn.Linear(in_features, out_features, bias=True)

                # Get new layer's parameters and replace with GPU versions
                new_params = new_layer.parameters()
                new_params[0] = weight.to_gpu(device_id)
                new_params[1] = bias.to_gpu(device_id)
                # Note: This won't work directly since parameters() returns copies
                # We need to actually modify the layer's internal state
                new_layers.append(layer)  # Keep original for now
            else:
                new_layers.append(layer)
        else:
            new_layers.append(layer)

    return model  # Return original for now


def benchmark():
    # Large synthetic dataset
    num_samples = 8192
    features = 4096
    classes = 100

    print("Generating dataset...")
    X = fnn.randn([num_samples, features])
    y = fnn.randint([num_samples], 0, classes)

    # Large model to ensure compute bound (not PCIe bound)
    print("Building model...")
    model = fnn.models.MLP(
        input_dim=features, hidden_dims=[2048, 2048, 1024], output_dim=classes
    )

    epochs = 5

    # === 1 GPU Benchmark (using 1080 Ti - faster GPU) ===
    print("\n" + "=" * 50)
    print("1 GPU BENCHMARK (GTX 1080 Ti)")
    print("=" * 50)
    # Create fresh model for 1-GPU test
    model_1gpu = fnn.models.MLP(
        input_dim=features, hidden_dims=[2048, 2048, 1024], output_dim=classes
    )
    opt_1 = fnn.Adam(model_1gpu.parameters(), lr=1e-3)

    start_time = time.time()
    for e in range(epochs):
        epoch_start = time.time()

        # Move data to GPU 0
        x_g = X.to_gpu(0)
        y_g = y.to_gpu(0)

        # Forward pass - model parameters will be moved to GPU on first operation
        pred = model_1gpu(x_g)
        loss = fnn.cross_entropy_loss(pred, y_g)

        opt_1.zero_grad()
        loss.backward()
        opt_1.step()

        print(
            f"  Epoch {e}: {time.time() - epoch_start:.3f}s - Loss: {loss.item():.4f}"
        )

    time_1gpu = time.time() - start_time

    # === 2 GPU DDP Benchmark (weighted split 70/30) ===
    print("\n" + "=" * 50)
    print("2 GPU DDP BENCHMARK (GTX 1080 Ti + GTX 1650)")
    print("Data split: 70% / 30% (weighted by GPU capability)")
    print("=" * 50)

    # Create separate model instances for each GPU (can't deepcopy Rust objects)
    model_gpu0 = fnn.models.MLP(
        input_dim=features, hidden_dims=[2048, 2048, 1024], output_dim=classes
    )
    model_gpu1 = fnn.models.MLP(
        input_dim=features, hidden_dims=[2048, 2048, 1024], output_dim=classes
    )

    # Create DataParallel with the replicas
    dp_model = DataParallel(
        [model_gpu0, model_gpu1], device_ids=[0, 1], weights=[0.7, 0.3]
    )
    opts = [fnn.Adam(dp_model.replicas[i].parameters(), lr=1e-3) for i in range(2)]

    start_time = time.time()
    for e in range(epochs):
        epoch_start = time.time()

        # Concurrently forward and backward with weighted split
        avg_loss = dp_model.forward_backward(X, y, fnn.cross_entropy_loss)

        # Fast Rust AllReduce - synchronize gradients
        dp_model.sync_gradients()

        # Step optimizers locally on each device
        for opt in opts:
            opt.step()
            opt.zero_grad()

        print(f"  Epoch {e}: {time.time() - epoch_start:.3f}s - Loss: {avg_loss:.4f}")

    time_2gpu = time.time() - start_time

    # === RESULTS ===
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"1 GPU (1080 Ti) Total Time: {time_1gpu:.2f}s")
    print(f"2 GPU DDP Total Time:       {time_2gpu:.2f}s")

    speedup = time_1gpu / time_2gpu
    print(f"\nSpeedup: {speedup:.2f}x")

    if speedup > 1.0:
        print(f"✓ DDP achieved {speedup:.2f}x speedup over single GPU")
    else:
        print(f"⚠ DDP slower than single GPU (speedup: {speedup:.2f}x)")
        print("  This can happen if:")
        print("  - Model is too small (PCIe overhead dominates)")
        print("  - 1650 GPU is significantly slower")
        print("  - Gradient synchronization overhead is high")


if __name__ == "__main__":
    print("Fastnn Distributed Data Parallel (DDP) Benchmark")
    print("-" * 50)
    benchmark()
