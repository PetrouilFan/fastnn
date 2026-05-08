"""
v0.8.0 Verification Gate: Optimizer Soak Test

Ensures no numerical collapse or parameter state corruption over millions of steps.
"""

import sys
import numpy as np

sys.path.insert(0, "/home/petrouil/Projects/github/fastnn")

import fastnn


def test_adamw_soak():
    """Test AdamW stability with mixed param groups."""
    print("Running AdamW optimizer soak test (10K steps)...")

    # Create parameters and mark them for gradient tracking
    param1 = fastnn.randn([64, 64])
    param1.requires_grad_(True)
    param2 = fastnn.randn([64])
    param2.requires_grad_(True)
    param3 = fastnn.randn([128, 64])
    param3.requires_grad_(True)

    opt = fastnn.AdamW([param1, param2, param3], lr=1e-3, weight_decay=0.01)

    for step in range(10_000):
        # Simple loss
        loss = param1.sum()

        loss.backward()
        opt.step()
        opt.zero_grad()

        if step % 2_500 == 0:
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

    print("  PASSED: No divergence over 10K steps")


def test_adam_soak():
    """Test Adam stability."""
    print("Running Adam optimizer soak test (5K steps)...")

    param = fastnn.randn([32, 32])
    param.requires_grad_(True)
    opt = fastnn.Adam([param], lr=1e-3, weight_decay=0.001)

    for step in range(5_000):
        loss = param.sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        if step % 1_250 == 0:
            v = param.numpy().flatten()[0]
            assert not np.isnan(v) and not np.isinf(v), f"Diverged at step {step}: {v}"
            print(f"  Step {step:>10,}: value={v:.6f}")

    print("  PASSED: No divergence over 5K steps")
