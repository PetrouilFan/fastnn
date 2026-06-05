from __future__ import annotations

from scripts.graph_memory_profile import build_memory_profile


def test_build_memory_profile_ranks_kernels_by_profiled_static_traffic() -> None:
    memory_stats = {
        "estimated_static_traffic_bytes": 3072,
        "kernel_read_bytes": 1024,
        "kernel_write_bytes": 2048,
        "memcpy_bytes": 512,
        "write_const_bytes": 256,
        "fill_bytes": 0,
        "top_kernels_by_count": [
            {"kernel": "conv2d_silu", "count": 2},
            {"kernel": "slice_f32", "count": 1},
        ],
    }
    profile_entries = [
        {"kernel_name": "conv2d_silu", "elapsed_ns": 2_000_000},
        {"kernel_name": "conv2d_silu", "elapsed_ns": 1_000_000},
        {"kernel_name": "slice_f32", "elapsed_ns": 500_000},
    ]

    profile = build_memory_profile(memory_stats, profile_entries)

    assert profile["summary"]["profiled_total_ms"] == 3.5
    assert profile["summary"]["profiled_kernel_static_bytes"] == 3072
    assert profile["summary"]["estimated_static_traffic_bytes"] == 3072
    assert profile["kernels"][0]["kernel_name"] == "conv2d_silu"
    assert profile["kernels"][0]["static_bytes"] == 2048
    assert profile["kernels"][0]["bytes_per_ms"] == 2048 / 3.0
    assert profile["kernels"][1]["kernel_name"] == "slice_f32"
    assert profile["kernels"][1]["suspected_memory_bound"] is True


def test_build_memory_profile_prefers_exact_per_kernel_static_bytes_when_available() -> None:
    memory_stats = {
        "estimated_static_traffic_bytes": 8192,
        "kernel_read_bytes": 4096,
        "kernel_write_bytes": 4096,
        "memcpy_bytes": 0,
        "write_const_bytes": 0,
        "fill_bytes": 0,
        "top_kernels_by_count": [
            {"kernel": "conv2d_silu", "count": 2, "read_bytes": 7000, "write_bytes": 1000},
            {"kernel": "slice_f32", "count": 1, "read_bytes": 64, "write_bytes": 64},
        ],
    }
    profile_entries = [
        {"kernel_name": "conv2d_silu", "elapsed_ns": 2_000_000},
        {"kernel_name": "conv2d_silu", "elapsed_ns": 2_000_000},
        {"kernel_name": "slice_f32", "elapsed_ns": 1_000_000},
    ]

    profile = build_memory_profile(memory_stats, profile_entries)

    conv = next(row for row in profile["kernels"] if row["kernel_name"] == "conv2d_silu")
    slice_row = next(row for row in profile["kernels"] if row["kernel_name"] == "slice_f32")
    assert conv["static_bytes"] == 8000
    assert conv["static_read_bytes"] == 7000
    assert conv["static_write_bytes"] == 1000
    assert slice_row["static_bytes"] == 128
    assert profile["summary"]["profiled_kernel_static_bytes"] == 8128


def test_build_memory_profile_accounts_unprofiled_write_and_copy_traffic() -> None:
    memory_stats = {
        "estimated_static_traffic_bytes": 4096,
        "kernel_read_bytes": 1024,
        "kernel_write_bytes": 1024,
        "memcpy_bytes": 1536,
        "write_const_bytes": 512,
        "fill_bytes": 0,
        "top_kernels_by_count": [{"kernel": "matmul", "count": 1}],
    }

    profile = build_memory_profile(memory_stats, [{"kernel_name": "matmul", "elapsed_ns": 1_000_000}])

    assert profile["summary"]["profiled_static_traffic_bytes"] == 2048
    assert profile["summary"]["unprofiled_static_traffic_bytes"] == 2048
    assert profile["unprofiled_static_traffic"] == [
        {"kind": "memcpy", "bytes": 1536},
        {"kind": "write_const", "bytes": 512},
    ]


def test_build_memory_profile_uses_profiled_copy_and_const_bytes_without_double_counting() -> None:
    memory_stats = {
        "estimated_static_traffic_bytes": 4096,
        "kernel_read_bytes": 1024,
        "kernel_write_bytes": 1024,
        "memcpy_bytes": 1536,
        "write_const_bytes": 512,
        "fill_bytes": 0,
        "top_kernels_by_count": [{"kernel": "matmul", "count": 1}],
    }

    profile = build_memory_profile(
        memory_stats,
        [
            {"kernel_name": "matmul", "elapsed_ns": 1_000_000},
            {"kernel_name": "memcopy", "elapsed_ns": 500_000},
            {"kernel_name": "write_const", "elapsed_ns": 250_000},
        ],
    )

    assert profile["summary"]["profiled_static_traffic_bytes"] == 4096
    assert profile["summary"]["profiled_kernel_static_bytes"] == 2048
    assert profile["summary"]["unprofiled_static_traffic_bytes"] == 0
    assert profile["unprofiled_static_traffic"] == []
    assert {row["kernel_name"]: row["static_bytes"] for row in profile["kernels"]} == {
        "matmul": 2048,
        "memcopy": 1536,
        "write_const": 512,
    }
