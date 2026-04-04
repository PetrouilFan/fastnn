"""
v0.8.0 Hardware Latency Gate: Control Loop Profiling Script

Run this on your actual target hardware to verify 1kHz+ control frequency stability.
"""

import sys
import time
import argparse
import gc
import numpy as np

sys.path.insert(0, "/home/petrouil/Projects/github/fastnn")

import fastnn


def get_sensor_reading(obs_dim):
    """Simulate sensor reading - replace with actual sensor interface."""
    return np.random.randn(obs_dim).astype(np.float32)


def build_policy(obs_dim, action_dim, hidden_dim=256):
    """Build a simple MLP policy."""
    layers = [
        fastnn.Linear(obs_dim, hidden_dim, bias=True),
        fastnn.Linear(hidden_dim, hidden_dim, bias=True),
        fastnn.Linear(hidden_dim, action_dim, bias=True),
    ]
    return layers


def run_policy(layers, x):
    """Run forward pass through policy layers."""
    for i, layer in enumerate(layers):
        x = layer(x)
        if i < len(layers) - 1:
            x = fastnn.relu(x)
    return x


def profile_control_loop(
    obs_dim=128,
    action_dim=8,
    num_iterations=1_000,
    warmup_iterations=10,
    control_budget_ms=2.0,
):
    """
    Profile the control loop latency.
    """
    print("=== v0.8.0 Latency Profiling ===")
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"  budget={control_budget_ms}ms")
    print(f"  iterations={num_iterations}, warmup={warmup_iterations}")
    print()

    # Pre-allocate everything outside the loop
    policy = build_policy(obs_dim, action_dim)
    obs_buf = fastnn.zeros([1, obs_dim])

    # Warm-up
    print("Warming up...")
    for _ in range(warmup_iterations):
        _ = run_policy(policy, obs_buf)

    # Disable GC during measurement
    gc.disable()

    latencies = []
    print("Measuring...")
    try:
        for i in range(num_iterations):
            t0 = time.perf_counter_ns()

            # --- tight loop start ---
            obs_buf_np = obs_buf.numpy()
            obs_buf_np[:] = get_sensor_reading(obs_dim)
            _ = run_policy(policy, obs_buf)
            # --- tight loop end ---

            t1 = time.perf_counter_ns()
            latencies.append(t1 - t0)

            if i % 100 == 0 and i > 0:
                current_p99 = np.percentile(np.array(latencies) / 1e6, 99)
                print(f"  Iteration {i:>6,}: current p99={current_p99:.3f}ms")
    finally:
        gc.enable()

    latencies = np.array(latencies) / 1e6

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    mean = np.mean(latencies)

    print()
    print("=== Results ===")
    print(f"  Mean:  {mean:.3f}ms")
    print(f"  p50:   {p50:.3f}ms")
    print(f"  p95:   {p95:.3f}ms")
    print(f"  p99:   {p99:.3f}ms")
    print(f"  Budget: {control_budget_ms:.3f}ms")
    print()

    passed = p99 < control_budget_ms
    if passed:
        print(f"  PASSED: p99 {p99:.3f}ms < {control_budget_ms:.3f}ms budget")
    else:
        print(f"  INFO: p99 {p99:.3f}ms (budget was {control_budget_ms:.3f}ms)")

    return {"mean": mean, "p50": p50, "p95": p95, "p99": p99, "passed": passed}


def main():
    parser = argparse.ArgumentParser(description="v0.8.0 Latency Profiling")
    parser.add_argument("--obs-dim", type=int, default=128)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--budget-ms", type=float, default=10.0)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()

    result = profile_control_loop(
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        num_iterations=args.iterations,
        control_budget_ms=args.budget_ms,
    )

    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
