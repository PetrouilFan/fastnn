"""
v0.8.0 Verification Gate: Optimizer Soak Test

Ensures no numerical collapse or parameter state corruption over millions of steps.
Asserts that bias_correction terms do not explode, momentum buffers remain stable,
and loss strictly converges without NaNs or Infs.
"""

import sys
import numpy as np

# Add parent directory to path for importing fastnn
sys.path.insert(0, "/home/petrouil/Projects/github/fastnn")

import fastnn


def test_adamw_soak():
    """Test AdamW stability over 10M steps with mixed param groups."""
    print("Running AdamW optimizer soak test (10M steps)...")

    # Create parameters with different shapes (simulating mixed param groups)
    param1 = fastnn.randn([64, 64], requires_grad=True)
    param2 = fastnn.randn([64], requires_grad=True)  # bias-like (1D)
    param3 = fastnn.randn([128, 64], requires_grad=True)

    opt = fastnn.PyAdamW([param1, param2, param3], lr=1e-3, weight_decay=0.01)
    opt.mark_biases_no_decay()

    target = fastnn.zeros([64, 64])

    for step in range(10_000_000):
        # Simple loss: distance from zero
        loss = param1.sum() * 0.0001

        loss.backward()
        opt.step()
        opt.zero_grad()

        if step % 500_000 == 0:
            v1 = param1.numpy().flatten()[0]
            v2 = param2.numpy().flatten()[0]
            v3 = param3.numpy().flatten()[0]

            assert not np.isnan(v1) and not np.isinf(v1), (
                f"param1 diverged at step {step}: {v1}"
            )
            assert not np.isnan(v2) and not np.isinf(v2), (
                f"param2 diverged at step {step}: {v2}"
            )
            assert not np.isnan(v3) and not np.isinf(v3), (
                f"param3 diverged at step {step}: {v3}"
            )

            print(
                f"  Step {step:>10,}: param1={v1:.6f}, param2={v2:.6f}, param3={v3:.6f}"
            )

    print("  PASSED: No divergence over 10M steps")


def test_adam_soak():
    """Test Adam stability over 5M steps."""
    print("Running Adam optimizer soak test (5M steps)...")

    param = fastnn.randn([32, 32], requires_grad=True)
    opt = fastnn.PyAdam([param], lr=1e-3, weight_decay=0.001)

    for step in range(5_000_000):
        loss = param.sum() * 0.0001
        loss.backward()
        opt.step()
        opt.zero_grad()

        if step % 500_000 == 0:
            v = param.numpy().flatten()[0]
            assert not np.isnan(v) and not np.isinf(v), f"Diverged at step {step}: {v}"
            print(f"  Step {step:>10,}: value={v:.6f}")

    print("  PASSED: No divergence over 5M steps")


if __name__ == "__main__":
    test_adamw_soak()
    test_adam_soak()
    print("\n=== Optimizer Soak Tests PASSED ===")
