import fastnn as fnn
import time
import statistics


def benchmark(name, fn, warmup=10, iterations=100):
    # Warmup
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


print("Running benchmarks...")
print("=" * 70)

# Add
for size in [(100, 100), (1000, 1000)]:
    a = fnn.rand(list(size))
    b = fnn.rand(list(size))
    mean, _ = benchmark("add", lambda: a + b)
    print(f"Add {size[0]}x{size[1]}: {mean:.1f}μs")

# Mul
for size in [(100, 100), (1000, 1000)]:
    a = fnn.rand(list(size))
    b = fnn.rand(list(size))
    mean, _ = benchmark("mul", lambda: a * b)
    print(f"Mul {size[0]}x{size[1]}: {mean:.1f}μs")

# ReLU
for size in [(100, 100), (1000, 1000)]:
    x = fnn.rand(list(size))
    mean, _ = benchmark("relu", lambda: fnn.relu(x))
    print(f"ReLU {size[0]}x{size[1]}: {mean:.1f}μs")

# FusedAddReLU
for size in [(100, 100), (1000, 1000)]:
    a = fnn.rand(list(size))
    b = fnn.rand(list(size))
    mean, _ = benchmark("fused_add_relu", lambda: fnn.fused_add_relu(a, b))
    print(f"FusedAddReLU {size[0]}x{size[1]}: {mean:.1f}μs")

# MatMul
for sizes in [(128, 256, 256, 128), (256, 512, 512, 256), (512, 1024, 1024, 512)]:
    m, k, k2, n = sizes
    a = fnn.rand([m, k])
    b = fnn.rand([k, n])
    mean, _ = benchmark("matmul", lambda: a @ b)
    print(f"MatMul {m}x{k}x{n}: {mean:.1f}μs")

# GELU
for size in [(100, 100), (1000, 1000)]:
    x = fnn.rand(list(size))
    mean, _ = benchmark("gelu", lambda: fnn.gelu(x))
    print(f"GELU {size[0]}x{size[1]}: {mean:.1f}μs")

# Sigmoid
for size in [(100, 100), (1000, 1000)]:
    x = fnn.rand(list(size))
    mean, _ = benchmark("sigmoid", lambda: fnn.sigmoid(x))
    print(f"Sigmoid {size[0]}x{size[1]}: {mean:.1f}μs")

# Tanh
for size in [(100, 100), (1000, 1000)]:
    x = fnn.rand(list(size))
    mean, _ = benchmark("tanh", lambda: fnn.tanh(x))
    print(f"Tanh {size[0]}x{size[1]}: {mean:.1f}μs")

# Linear
for config in [(32, 256, 512), (32, 512, 1024), (128, 256, 512)]:
    batch_size, in_features, out_features = config
    linear = fnn.Linear(in_features, out_features)
    x = fnn.rand([batch_size, in_features])
    mean, _ = benchmark("linear", lambda: linear(x))
    print(f"Linear {batch_size}x{in_features}x{out_features}: {mean:.1f}μs")

# Sum
size = (1000, 1000)
x = fnn.rand(list(size))
mean, _ = benchmark("sum", lambda: fnn.sum(x, 1))
print(f"Sum {size[0]}x{size[1]}: {mean:.1f}μs")

# Mean
x = fnn.rand(list(size))
mean, _ = benchmark("mean", lambda: fnn.mean(x, 1))
print(f"Mean {size[0]}x{size[1]}: {mean:.1f}μs")

# Max
x = fnn.rand(list(size))
mean, _ = benchmark("max", lambda: fnn.max(x, 1))
print(f"Max {size[0]}x{size[1]}: {mean:.1f}μs")

print("=" * 70)
print("Done!")
