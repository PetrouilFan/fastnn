#!/usr/bin/env python3
"""Benchmark backward passes to establish baselines before optimization."""

import time
import fastnn as nn
import numpy as np

def benchmark(name, setup_fn, iters=30, warmup=5):
    """Generic benchmark helper.
    setup_fn should return a function that runs the backward pass.
    """
    # Warmup
    for _ in range(warmup):
        fn = setup_fn()
        fn()

    times = []
    for _ in range(iters):
        fn = setup_fn()
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    median_time = np.median(times)
    fps = 1000.0 / median_time
    print(f"{name:50s}: {median_time:7.2f} ms  ({fps:6.1f} FPS)")
    return median_time

def main():
    print("=" * 80)
    print("FastNN Backward Pass Benchmarks (BASELINE)")
    print("=" * 80)

    np.random.seed(42)

    # ============================================================
    # 1. SoftmaxBackward - Critical for transformers (attention)
    # ============================================================
    print("\n--- SoftmaxBackward ---")

    configs = [
        (1, 8, 64, 64),      # Small attention
        (1, 16, 128, 128),    # Medium attention
        (4, 16, 256, 256),    # Large attention
    ]

    for batch, heads, seq, seq2 in configs:
        print(f"\nConfig: {batch}x{heads}x{seq}x{seq2} (attention logits)")

        def make_softmax_backward(batch=batch, heads=heads, seq=seq, seq2=seq2):
            logits_data = np.random.randn(batch, heads, seq, seq2).flatten().tolist()
            logits = nn.tensor(logits_data, [batch, heads, seq, seq2])
            logits.requires_grad_(True)
            grad_out_data = np.random.randn(batch, heads, seq, seq2).flatten().tolist()
            grad_output = nn.tensor(grad_out_data, [batch, heads, seq, seq2])

            def backward():
                y = nn.softmax(logits, dim=-1)
                y.backward(grad_output)
                return logits.grad
            return backward

        benchmark("SoftmaxBackward", make_softmax_backward)

    # ============================================================
    # 2. SiLUBackward - Used in LLaMA
    # ============================================================
    print("\n\n--- SiLUBackward ---")

    configs = [
        (1, 1024),
        (1, 4096),
        (4, 4096),
    ]

    for batch, features in configs:
        print(f"\nConfig: {batch}x{features}")

        def make_silu_backward(batch=batch, features=features):
            x_data = np.random.randn(batch, features).flatten().tolist()
            x = nn.tensor(x_data, [batch, features])
            x.requires_grad_(True)
            grad_out_data = np.random.randn(batch, features).flatten().tolist()
            grad_output = nn.tensor(grad_out_data, [batch, features])

            def backward():
                y = nn.silu(x)
                y.backward(grad_output)
                return x.grad
            return backward

        benchmark("SiLUBackward", make_silu_backward)

    # ============================================================
    # 3. GELUBackward - Already fused (baseline comparison)
    # ============================================================
    print("\n\n--- GELUBackward (fused - for comparison) ---")

    configs = [
        (1, 1024),
        (1, 4096),
        (4, 4096),
    ]

    for batch, features in configs:
        print(f"\nConfig: {batch}x{features}")

        def make_gelu_backward(batch=batch, features=features):
            x_data = np.random.randn(batch, features).flatten().tolist()
            x = nn.tensor(x_data, [batch, features])
            x.requires_grad_(True)
            grad_out_data = np.random.randn(batch, features).flatten().tolist()
            grad_output = nn.tensor(grad_out_data, [batch, features])

            def backward():
                y = nn.gelu(x)
                y.backward(grad_output)
                return x.grad
            return backward

        benchmark("GELUBackward (fused)", make_gelu_backward)

    # ============================================================
    # 4. HardswishBackward
    # ============================================================
    print("\n\n--- HardswishBackward ---")

    configs = [
        (1, 1024),
        (1, 4096),
        (4, 4096),
    ]

    for batch, features in configs:
        print(f"\nConfig: {batch}x{features}")

        def make_hardswish_backward(batch=batch, features=features):
            x_data = np.random.randn(batch, features).flatten().tolist()
            x = nn.tensor(x_data, [batch, features])
            x.requires_grad_(True)
            grad_out_data = np.random.randn(batch, features).flatten().tolist()
            grad_output = nn.tensor(grad_out_data, [batch, features])

            def backward():
                y = nn.Hardswish()(x)
                y.backward(grad_output)
                return x.grad
            return backward

        benchmark("HardswishBackward", make_hardswish_backward)

    # ============================================================
    # 5. EluBackward
    # ============================================================
    print("\n\n--- EluBackward ---")

    configs = [
        (1, 1024),
        (1, 4096),
        (4, 4096),
    ]

    for batch, features in configs:
        print(f"\nConfig: {batch}x{features}")

        def make_elu_backward(batch=batch, features=features):
            x_data = np.random.randn(batch, features).flatten().tolist()
            x = nn.tensor(x_data, [batch, features])
            x.requires_grad_(True)
            grad_out_data = np.random.randn(batch, features).flatten().tolist()
            grad_output = nn.tensor(grad_out_data, [batch, features])

            def backward():
                y = nn.Elu()(x)
                y.backward(grad_output)
                return x.grad
            return backward

        benchmark("EluBackward", make_elu_backward)

    # ============================================================
    # 6. LeakyReLUBackward
    # ============================================================
    print("\n\n--- LeakyReLUBackward ---")

    configs = [
        (1, 1024),
        (1, 4096),
        (4, 4096),
    ]

    for batch, features in configs:
        print(f"\nConfig: {batch}x{features}")

        def make_leaky_relu_backward(batch=batch, features=features):
            x_data = np.random.randn(batch, features).flatten().tolist()
            x = nn.tensor(x_data, [batch, features])
            x.requires_grad_(True)
            grad_out_data = np.random.randn(batch, features).flatten().tolist()
            grad_output = nn.tensor(grad_out_data, [batch, features])

            def backward():
                y = nn.LeakyReLU()(x)
                y.backward(grad_output)
                return x.grad
            return backward

        benchmark("LeakyReLUBackward", make_leaky_relu_backward)

    # ============================================================
    # 7. CrossEntropyBackward
    # ============================================================
    print("\n\n--- CrossEntropyBackward ---")

    configs = [
        (32, 1000),
        (64, 1000),
        (128, 1000),
    ]

    for batch, classes in configs:
        print(f"\nConfig: batch={batch}, classes={classes}")

        def make_cross_entropy_backward(batch=batch, classes=classes):
            logits_data = np.random.randn(batch, classes).flatten().tolist()
            logits = nn.tensor(logits_data, [batch, classes])
            logits.requires_grad_(True)
            targets_data = np.random.randint(0, classes, batch).tolist()
            targets = nn.tensor(targets_data, [batch])

            def backward():
                loss = nn.cross_entropy_loss(logits, targets)
                loss.backward()
                return logits.grad
            return backward

        benchmark("CrossEntropyBackward", make_cross_entropy_backward)

    print("\n" + "=" * 80)
    print("BASELINE BENCHMARKS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
