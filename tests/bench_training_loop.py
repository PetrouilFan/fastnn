"""
Training loop benchmark - measures forward + backward + optimizer step
Compares fastnn vs PyTorch to track optimization progress.
"""

import time
import statistics
import torch
import fastnn as fnn
import numpy as np


def benchmark(name, fn, warmup=5, iterations=20):
    """Run a benchmark and return mean/std in microseconds."""
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append((end - start) * 1e6)  # Convert to microseconds

    mean_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0
    return mean_time, std_time


def create_model_fnn(input_dim, hidden_dims, output_dim):
    """Create a simple MLP in fastnn."""
    layers = [fnn.Linear(input_dim, hidden_dims[0]), fnn.ReLU()]
    for i in range(len(hidden_dims) - 1):
        layers.append(fnn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        layers.append(fnn.ReLU())
    layers.append(fnn.Linear(hidden_dims[-1], output_dim))
    return fnn.Sequential(*layers)


def create_model_torch(input_dim, hidden_dims, output_dim):
    """Create a simple MLP in PyTorch."""
    layers = [torch.nn.Linear(input_dim, hidden_dims[0]), torch.nn.ReLU()]
    for i in range(len(hidden_dims) - 1):
        layers.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(hidden_dims[-1], output_dim))
    return torch.nn.Sequential(*layers)


def run_training_step_fnn(model, optimizer, x, y):
    """Single training step for fastnn."""
    optimizer.zero_grad()
    pred = model(x)
    loss = fnn.cross_entropy_loss(pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def run_training_step_torch(model, optimizer, x, y):
    """Single training step for PyTorch."""
    optimizer.zero_grad()
    pred = model(x)
    loss = torch.nn.functional.cross_entropy(pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def benchmark_linear_layer():
    """Benchmark single Linear layer forward + backward."""
    print("\n" + "=" * 70)
    print("BENCHMARK: Single Linear Layer (forward + backward)")
    print("=" * 70)

    configs = [
        (256, 512),
        (512, 1024),
        (1024, 2048),
        (2048, 4096),
    ]
    batch_size = 32

    for in_feat, out_feat in configs:
        # Fastnn
        linear_fnn = fnn.Linear(in_feat, out_feat)
        x_fnn = fnn.randn([batch_size, in_feat])
        target_fnn = fnn.randint(0, out_feat, [batch_size])

        # PyTorch
        linear_torch = torch.nn.Linear(in_feat, out_feat)
        x_torch = torch.rand(batch_size, in_feat)
        target_torch = torch.randint(0, out_feat, (batch_size,))

        # Fastnn benchmark
        fnn_mean, fnn_std = benchmark(
            "fastnn",
            lambda: [
                linear_fnn(x_fnn),
                fnn.cross_entropy_loss(linear_fnn(x_fnn), target_fnn).backward(),
            ],
        )

        # PyTorch benchmark
        torch_mean, torch_std = benchmark(
            "torch",
            lambda: [
                linear_torch(x_torch),
                torch.nn.functional.cross_entropy(
                    linear_torch(x_torch), target_torch
                ).backward(),
            ],
        )

        ratio = torch_mean / fnn_mean if fnn_mean > 0 else 0
        print(f"Linear {in_feat} -> {out_feat} (batch={batch_size}):")
        print(f"  Fastnn: {fnn_mean:.1f} ± {fnn_std:.1f} μs")
        print(f"  PyTorch: {torch_mean:.1f} ± {torch_std:.1f} μs")
        print(f"  Ratio (torch/fastnn): {ratio:.2f}x")


def benchmark_optimizer_step():
    """Benchmark optimizer step only (no forward/backward)."""
    print("\n" + "=" * 70)
    print("BENCHMARK: Optimizer Step Only")
    print("=" * 70)

    configs = [
        (100, 1024),  # 100K params
        (500, 1024),  # 500K params
        (1000, 1024),  # 1M params
    ]

    for batch_size, in_feat in configs:
        numel = batch_size * in_feat

        # Create tensors with gradients
        param_fnn = fnn.randn([batch_size, in_feat]).requires_grad_(True)
        grad_fnn = fnn.randn([batch_size, in_feat])
        param_fnn.inner.autograd_meta = None
        param_fnn = fnn.randn([batch_size, in_feat])

        # Attach gradient manually
        param_fnn.inner.autograd_meta = None
        _ = param_fnn.clone()
        param_fnn = fnn.randn([batch_size, in_feat])

        # Fastnn optimizer
        params_fnn = [
            fnn.randn([batch_size, in_feat]).requires_grad_(True) for _ in range(3)
        ]
        opt_fnn = fnn.AdamW(params_fnn, lr=0.001, weight_decay=0.01)

        # PyTorch optimizer
        params_torch = [
            torch.randn(batch_size, in_feat, requires_grad=True) for _ in range(3)
        ]
        opt_torch = torch.optim.AdamW(params_torch, lr=0.001, weight_decay=0.01)

        def step_fnn():
            for p in params_fnn:
                if p.grad() is not None:
                    p.inner.autograd_meta = None
            grad = fnn.randn([batch_size, in_feat])
            for p in params_fnn:
                p.inner.autograd_meta = None
            # Simple step
            pass

        # For fair comparison, just measure the optimizer step with pre-computed grads
        params_fnn2 = [
            fnn.randn([batch_size, in_feat]).requires_grad_(True) for _ in range(10)
        ]
        opt_fnn2 = fnn.AdamW(params_fnn2, lr=0.001)

        for p in params_fnn2:
            p.inner.autograd_meta = None  # Clear autograd to avoid backward overhead

        params_torch2 = [
            torch.randn(batch_size, in_feat, requires_grad=True) for _ in range(10)
        ]
        opt_torch2 = torch.optim.AdamW(params_torch2, lr=0.001)

        # Run one step to warm up
        for p, g in zip(
            params_fnn2, [fnn.randn([batch_size, in_feat]) for _ in range(10)]
        ):
            pass
        opt_fnn2.step()
        opt_torch2.step()

        # Re-create for actual benchmark
        params_fnn3 = [
            fnn.randn([batch_size, in_feat]).requires_grad_(True) for _ in range(10)
        ]
        opt_fnn3 = fnn.AdamW(params_fnn3, lr=0.001)

        params_torch3 = [
            torch.randn(batch_size, in_feat, requires_grad=True) for _ in range(10)
        ]
        opt_torch3 = torch.optim.AdamW(params_torch3, lr=0.001)

        print(f"Optimizer with {10 * batch_size * in_feat:,} parameters:")
        print(f"  (Measuring full step including parameter updates)")

        # This is complex to benchmark fairly, so we skip for now
        print(f"  Note: Individual op benchmarks in bench_comparison.py")


def benchmark_mlp_training():
    """Benchmark full MLP training: forward + backward + optimizer."""
    print("\n" + "=" * 70)
    print("BENCHMARK: MLP Training Loop (forward + backward + optimizer)")
    print("=" * 70)

    configs = [
        (784, [256, 256], 10, 32),  # Small MLP, MNIST-like
        (784, [512, 512, 256], 10, 64),  # Medium MLP
        (128, [256, 256, 128], 10, 32),  # Small transformer-like
    ]

    for input_dim, hidden_dims, output_dim, batch_size in configs:
        total_params = input_dim * hidden_dims[0]
        for i in range(len(hidden_dims) - 1):
            total_params += hidden_dims[i] * hidden_dims[i + 1]
        total_params += hidden_dims[-1] * output_dim
        total_params += output_dim  # bias terms

        # Create models
        model_fnn = create_model_fnn(input_dim, hidden_dims, output_dim)
        model_torch = create_model_torch(input_dim, hidden_dims, output_dim)

        # Create optimizer
        opt_fnn = fnn.AdamW(model_fnn.parameters(), lr=0.001)
        opt_torch = torch.optim.AdamW(model_torch.parameters(), lr=0.001)

        # Create data
        x_fnn = fnn.randn([batch_size, input_dim])
        y_fnn = fnn.randint(0, output_dim, [batch_size])

        x_torch = torch.rand(batch_size, input_dim)
        y_torch = torch.randint(0, output_dim, (batch_size,))

        def train_fnn():
            opt_fnn.zero_grad()
            pred = model_fnn(x_fnn)
            loss = fnn.cross_entropy_loss(pred, y_fnn)
            loss.backward()
            opt_fnn.step()
            return loss.item()

        def train_torch():
            opt_torch.zero_grad()
            pred = model_torch(x_torch)
            loss = torch.nn.functional.cross_entropy(pred, y_torch)
            loss.backward()
            opt_torch.step()
            return loss.item()

        # Benchmark
        fnn_mean, fnn_std = benchmark("fastnn", train_fnn)
        torch_mean, torch_std = benchmark("torch", train_torch)

        ratio = torch_mean / fnn_mean if fnn_mean > 0 else 0
        print(
            f"\nMLP {input_dim} -> {hidden_dims} -> {output_dim} (batch={batch_size}, {total_params:,} params):"
        )
        print(f"  Fastnn: {fnn_mean:.1f} ± {fnn_std:.1f} μs")
        print(f"  PyTorch: {torch_mean:.1f} ± {torch_std:.1f} μs")
        print(f"  Ratio (torch/fastnn): {ratio:.2f}x")

        if ratio > 1:
            print(f"  -> Fastnn is {ratio:.2f}x FASTER than PyTorch!")
        else:
            print(f"  -> Fastnn is {1 / ratio:.2f}x SLOWER than PyTorch")


def benchmark_elementwise_ops():
    """Benchmark elementwise operations (baseline)."""
    print("\n" + "=" * 70)
    print("BENCHMARK: Elementwise Operations")
    print("=" * 70)

    sizes = [
        (100, 100),  # 10K
        (1000, 1000),  # 1M
    ]

    ops = [
        ("Add", lambda a, b: a + b),
        ("Mul", lambda a, b: a * b),
        ("ReLU", lambda x: fnn.relu(x)),
        ("GELU", lambda x: fnn.gelu(x)),
    ]

    torch_ops = [
        ("Add", lambda a, b: a + b),
        ("Mul", lambda a, b: a * b),
        ("ReLU", lambda x: torch.nn.functional.relu(x)),
        ("GELU", lambda x: torch.nn.functional.gelu(x)),
    ]

    for size in sizes:
        print(f"\nSize: {size[0]}x{size[1]} ({size[0] * size[1]:,} elements)")

        for (name, op_fnn), (_, op_torch) in zip(ops, torch_ops):
            a_fnn = fnn.rand(list(size))
            b_fnn = fnn.rand(list(size))
            x_fnn = fnn.rand(list(size))

            a_torch = torch.rand(size)
            b_torch = torch.rand(size)
            x_torch = torch.rand(size)

            fnn_mean, _ = benchmark(
                f"fastnn/{name}",
                lambda: (
                    op_fnn(a_fnn, b_fnn) if name in ["Add", "Mul"] else op_fnn(x_fnn)
                ),
            )
            torch_mean, _ = benchmark(
                f"torch/{name}",
                lambda: (
                    op_torch(a_torch, b_torch)
                    if name in ["Add", "Mul"]
                    else op_torch(x_torch)
                ),
            )

            ratio = torch_mean / fnn_mean if fnn_mean > 0 else 0
            print(
                f"  {name}: fastnn={fnn_mean:.1f}μs, torch={torch_mean:.1f}μs, ratio={ratio:.2f}x"
            )


def main():
    print("=" * 70)
    print("FASTNN PERFORMANCE BENCHMARK SUITE")
    print("Comparing fastnn vs PyTorch for training workloads")
    print("=" * 70)

    benchmark_elementwise_ops()
    benchmark_linear_layer()
    benchmark_mlp_training()

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print("\nRun with: python tests/bench_training_loop.py")
    print(
        "Run pytest benchmarks with: pytest tests/bench_comparison.py --benchmark-only"
    )


if __name__ == "__main__":
    main()
