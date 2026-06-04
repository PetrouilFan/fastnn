"""Smoke tests for `AotExecutor.prepared_stats()` and the new
constant-arena / static-weight-binding keys added in mission 015.

These tests build a tiny AOT executor from hand-written ONNX-style
nodes (the same harness `test_aot_executor_profile.py` uses) and
introspect the returned stats dict. They require the
`prepared-plan` Rust feature to be enabled at build time.
"""

import pytest


def test_prepared_stats_has_arena_keys():
    import fastnn as fnn

    nodes = [
        {
            "name": "relu1",
            "op_type": "Relu",
            "inputs": "x",
            "outputs": "y",
        }
    ]
    executor = fnn.AotExecutor(
        nodes,
        {},
        ["x"],
        ["y"],
        input_shapes={"x": [1, 4]},
    )

    stats = executor.prepared_stats()
    assert isinstance(stats, dict)
    # Baseline counters from earlier waves.
    for key in ("total", "generic", "conv2d", "matmul"):
        assert key in stats, f"missing baseline key: {key}"
    # New introspection keys added in mission 015.
    for key in (
        "static_weight_bindings",
        "constant_arena_entries",
        "constant_arena_bytes",
        "packed_fp32_conv_candidates",
        "packed_fp32_conv_candidate_flops",
    ):
        assert key in stats, f"missing arena key: {key}"
        assert isinstance(stats[key], int), f"{key} should be int"
        assert stats[key] >= 0, f"{key} should be non-negative"

    # A plain Relu plan has no Conv2d / MatMul, so the three new keys
    # must all be zero. The auto-attached arena is empty in that
    # case and no static-weight bindings are recorded.
    assert stats["conv2d"] == 0
    assert stats["matmul"] == 0
    assert stats["static_weight_bindings"] == 0
    assert stats["constant_arena_entries"] == 0
    assert stats["constant_arena_bytes"] == 0
    assert stats["packed_fp32_conv_candidates"] == 0
    assert stats["packed_fp32_conv_candidate_flops"] == 0
    # `total` should still match the baseline counter.
    assert stats["total"] == stats["generic"] + stats["conv2d"] + stats["matmul"]
