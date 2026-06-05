from __future__ import annotations

from scripts.graph_memory_profile import build_memory_profile, build_profile_delta, build_profile_payload


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


def _profile_payload(
    *,
    total_ms: float,
    kernel_static_bytes: int,
    estimated_static_bytes: int,
    kernels: list[dict[str, object]],
    unprofiled: list[dict[str, object]] | None = None,
    write_const_bytes: int = 0,
) -> dict[str, object]:
    return {
        "summary": {
            "profiled_total_ms": total_ms,
            "estimated_static_traffic_bytes": estimated_static_bytes,
            "profiled_kernel_static_bytes": kernel_static_bytes,
        },
        "kernels": kernels,
        "unprofiled_static_traffic": unprofiled or [],
        "memory_stats": {"write_const_bytes": write_const_bytes},
    }


def test_build_profile_delta_reports_removed_write_const_entries() -> None:
    default = _profile_payload(
        total_ms=3.0,
        kernel_static_bytes=4096,
        estimated_static_bytes=16_384,
        write_const_bytes=12_288,
        kernels=[{"kernel_name": "write_const", "profile_count": 3, "total_ms": 1.5, "static_bytes": 12_288}],
    )
    prepared = _profile_payload(
        total_ms=2.0,
        kernel_static_bytes=4096,
        estimated_static_bytes=8192,
        write_const_bytes=4096,
        kernels=[{"kernel_name": "write_const", "profile_count": 1, "total_ms": 0.5, "static_bytes": 4096}],
    )

    delta = build_profile_delta(default, prepared)

    assert delta["summary"]["write_const_count_delta"] == -2
    assert delta["summary"]["write_const_static_bytes_delta"] == -8192
    assert delta["summary"]["write_const_total_ms_delta"] == -1.0
    write_const = next(row for row in delta["kernels"] if row["kernel_name"] == "write_const")
    assert write_const["count_delta"] == -2
    assert write_const["removed"] is False
    assert write_const["added"] is False


def test_build_profile_delta_marks_removed_and_added_kernels() -> None:
    default = _profile_payload(
        total_ms=4.0,
        kernel_static_bytes=4096,
        estimated_static_bytes=4096,
        kernels=[{"kernel_name": "slice_f32", "profile_count": 4, "total_ms": 4.0, "static_bytes": 4096}],
    )
    prepared = _profile_payload(
        total_ms=4.0,
        kernel_static_bytes=4096,
        estimated_static_bytes=4096,
        kernels=[{"kernel_name": "prepared_slice", "profile_count": 4, "total_ms": 4.0, "static_bytes": 4096}],
    )

    delta = build_profile_delta(default, prepared)

    rows = {row["kernel_name"]: row for row in delta["kernels"]}
    assert rows["slice_f32"]["removed"] is True
    assert rows["slice_f32"]["added"] is False
    assert rows["slice_f32"]["count_delta"] == -4
    assert rows["slice_f32"]["total_ms_delta"] == -4.0
    assert rows["prepared_slice"]["removed"] is False
    assert rows["prepared_slice"]["added"] is True
    assert rows["prepared_slice"]["count_delta"] == 4


def test_build_profile_delta_summarises_total_profile_time() -> None:
    default = _profile_payload(total_ms=10.0, kernel_static_bytes=12_000, estimated_static_bytes=20_000, kernels=[])
    prepared = _profile_payload(total_ms=6.0, kernel_static_bytes=8000, estimated_static_bytes=18_000, kernels=[])

    delta = build_profile_delta(default, prepared)

    assert delta["summary"]["profiled_total_ms_delta"] == -4.0
    assert delta["summary"]["profiled_kernel_static_bytes_delta"] == -4000
    assert delta["summary"]["estimated_static_traffic_bytes_delta"] == -2000


def test_build_profile_delta_uses_unprofiled_write_const_when_profiled_silences_it() -> None:
    default = _profile_payload(
        total_ms=1.0,
        kernel_static_bytes=0,
        estimated_static_bytes=12_288,
        write_const_bytes=12_288,
        kernels=[{"kernel_name": "write_const", "profile_count": 3, "total_ms": 1.5, "static_bytes": 12_288}],
    )
    prepared = _profile_payload(
        total_ms=1.0,
        kernel_static_bytes=0,
        estimated_static_bytes=4096,
        write_const_bytes=4096,
        kernels=[],
        unprofiled=[{"kind": "write_const", "bytes": 4096}],
    )

    delta = build_profile_delta(default, prepared)

    assert delta["summary"]["write_const_count_delta"] == -3
    assert delta["summary"]["write_const_static_bytes_delta"] == -8192
    assert delta["summary"]["unprofiled_write_const_bytes_delta"] == 4096


def test_build_profile_delta_apportions_write_const_bytes_when_memory_stats_are_shared() -> None:
    default = _profile_payload(
        total_ms=1.0,
        kernel_static_bytes=0,
        estimated_static_bytes=12_288,
        write_const_bytes=12_288,
        kernels=[{"kernel_name": "write_const", "profile_count": 3, "total_ms": 1.5, "static_bytes": 12_288}],
    )
    default["memory_stats"]["write_const_count"] = 3
    prepared = _profile_payload(
        total_ms=1.0,
        kernel_static_bytes=0,
        estimated_static_bytes=12_288,
        write_const_bytes=12_288,
        kernels=[{"kernel_name": "write_const", "profile_count": 1, "total_ms": 0.5, "static_bytes": 12_288}],
    )
    prepared["memory_stats"]["write_const_count"] = 3

    delta = build_profile_delta(default, prepared)

    assert delta["summary"]["write_const_count_delta"] == -2
    assert delta["summary"]["write_const_static_bytes_delta"] == -8192


def test_build_memory_profile_treats_memcpy_spelling_as_profiled_copy_traffic() -> None:
    memory_stats = {
        "estimated_static_traffic_bytes": 1536,
        "kernel_read_bytes": 0,
        "kernel_write_bytes": 0,
        "memcpy_bytes": 1536,
        "write_const_bytes": 0,
        "fill_bytes": 0,
        "top_kernels_by_count": [],
    }

    profile = build_memory_profile(memory_stats, [{"kernel_name": "memcpy", "elapsed_ns": 500_000}])

    assert profile["summary"]["profiled_static_traffic_bytes"] == 1536
    assert profile["summary"]["unprofiled_static_traffic_bytes"] == 0
    assert profile["unprofiled_static_traffic"] == []
    assert profile["kernels"][0]["kernel_name"] == "memcpy"
    assert profile["kernels"][0]["static_bytes"] == 1536


def test_build_profile_payload_preserves_single_profile_shape_without_prepared_entries() -> None:
    memory_stats = {
        "estimated_static_traffic_bytes": 1024,
        "kernel_read_bytes": 0,
        "kernel_write_bytes": 1024,
        "top_kernels_by_count": [{"kernel": "add_f32", "count": 1}],
    }

    payload = build_profile_payload(memory_stats, [{"kernel_name": "add_f32", "elapsed_ns": 1_000_000}])

    assert set(payload) == {"summary", "kernels", "unprofiled_static_traffic", "memory_stats"}
    assert payload["summary"]["profiled_total_ms"] == 1.0
    assert payload["kernels"][0]["kernel_name"] == "add_f32"


def test_build_profile_payload_embeds_default_prepared_and_delta_profiles() -> None:
    memory_stats = {
        "estimated_static_traffic_bytes": 12_288,
        "kernel_read_bytes": 0,
        "kernel_write_bytes": 0,
        "write_const_bytes": 12_288,
        "write_const_count": 3,
        "top_kernels_by_count": [],
    }
    default_entries = [
        {"kernel_name": "write_const", "elapsed_ns": 1_000_000},
        {"kernel_name": "write_const", "elapsed_ns": 2_000_000},
        {"kernel_name": "write_const", "elapsed_ns": 3_000_000},
    ]
    prepared_entries = [{"kernel_name": "write_const", "elapsed_ns": 500_000}]

    payload = build_profile_payload(memory_stats, default_entries, prepared_entries)

    assert set(payload) == {"default", "prepared", "delta"}
    assert payload["default"]["summary"]["profiled_total_ms"] == 6.0
    assert payload["prepared"]["summary"]["profiled_total_ms"] == 0.5
    assert payload["delta"]["summary"]["write_const_count_delta"] == -2
    assert payload["delta"]["summary"]["write_const_static_bytes_delta"] == -8192
