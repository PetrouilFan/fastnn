#!/usr/bin/env python3
"""Measure YOLO write_const / non-GEMM overhead from profile JSON exports.

Roadmap Task B2 preflight: quantify repeated constant materialisation before
attempting any default-path constant cache/preload change.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


KEYS = [
    "write_const",
    "conv2d_silu",
    "conv2d",
    "transpose_perm_f32",
    "pool_f32",
    "concat",
    "slice_f32",
    "sigmoid_f32",
    "softmax",
    "upsample_nearest2d",
    "add_f32",
    "memcopy",
]


def _run_once(args: argparse.Namespace, run_idx: int, tmp_dir: Path) -> dict[str, Any]:
    profile_json = tmp_dir / f"profile_{run_idx}.json"
    cmd = [
        sys.executable,
        "scripts/yolo_compare_fastnn_pytorch.py",
        "--profile",
        "--profile-top",
        str(args.profile_top),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--profile-json",
        str(profile_json),
    ]
    env = os.environ.copy()
    if args.openblas_threads is not None:
        env["OPENBLAS_NUM_THREADS"] = str(args.openblas_threads)
    if args.disable_openblas_conv_gemm:
        env["FASTNN_DISABLE_OPENBLAS_CONV_GEMM"] = "1"
    else:
        env.pop("FASTNN_DISABLE_OPENBLAS_CONV_GEMM", None)

    proc = subprocess.run(cmd, cwd=args.cwd, env=env, text=True, capture_output=True)
    if proc.returncode != 0:
        return {
            "ok": False,
            "run": run_idx,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
        }
    data = json.loads(profile_json.read_text())
    summary = {row["kernel_name"]: row for row in data.get("fastnn_profile_summary", [])}
    raw = data.get("fastnn_profile_raw", [])
    write_raw = [r for r in raw if r.get("kernel_name") == "write_const"]
    first_non_write = next((r.get("instruction_index") for r in raw if r.get("kernel_name") != "write_const"), None)
    totals = {key: float(summary.get(key, {}).get("total_ms", 0.0)) for key in KEYS}
    counts = {key: int(summary.get(key, {}).get("count", 0)) for key in KEYS}
    fastnn_mean = float(data["fastnn_speed"]["mean_ms"])
    write_total = totals["write_const"]
    non_conv_total = sum(v for k, v in totals.items() if k not in {"conv2d_silu", "conv2d"})
    return {
        "ok": True,
        "run": run_idx,
        "profile_json": str(profile_json),
        "fastnn_speed": data.get("fastnn_speed"),
        "fastnn_vs_pytorch": data.get("fastnn_vs_pytorch"),
        "totals_ms": totals,
        "counts": counts,
        "write_const_percent_of_fastnn_mean": write_total / fastnn_mean * 100.0 if fastnn_mean else 0.0,
        "non_conv_profile_total_ms": non_conv_total,
        "non_conv_percent_of_fastnn_mean": non_conv_total / fastnn_mean * 100.0 if fastnn_mean else 0.0,
        "first_non_write_instruction": first_non_write,
        "top_write_const_ms": sorted((r["elapsed_ns"] / 1_000_000.0 for r in write_raw), reverse=True)[:10],
    }


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--profile-top", type=int, default=20)
    ap.add_argument("--openblas-threads", type=int, default=2)
    ap.add_argument("--disable-openblas-conv-gemm", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    args.cwd = Path.cwd()

    tmp_dir = Path(tempfile.mkdtemp(prefix="fastnn-yolo-constant-overhead-"))
    rows = [_run_once(args, i, tmp_dir) for i in range(args.runs)]
    ok_rows = [r for r in rows if r.get("ok")]
    if not ok_rows:
        result = {"ok": False, "rows": rows}
    else:
        aggregate_totals = {
            key: {
                "mean_ms": _mean([r["totals_ms"][key] for r in ok_rows]),
                "count": ok_rows[0]["counts"].get(key, 0),
            }
            for key in KEYS
        }
        fastnn_means = [r["fastnn_speed"]["mean_ms"] for r in ok_rows]
        write_pcts = [r["write_const_percent_of_fastnn_mean"] for r in ok_rows]
        non_conv_pcts = [r["non_conv_percent_of_fastnn_mean"] for r in ok_rows]
        result = {
            "ok": all(r.get("ok") for r in rows),
            "config": {
                "runs": args.runs,
                "warmup": args.warmup,
                "iters": args.iters,
                "openblas_threads": args.openblas_threads,
                "disable_openblas_conv_gemm": args.disable_openblas_conv_gemm,
            },
            "fastnn_mean_ms": {
                "mean": _mean(fastnn_means),
                "min": min(fastnn_means),
                "max": max(fastnn_means),
            },
            "aggregate_totals": aggregate_totals,
            "write_const_percent_of_fastnn_mean": _mean(write_pcts),
            "non_conv_percent_of_fastnn_mean": _mean(non_conv_pcts),
            "rows": rows,
        }

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True))

    if result.get("ok"):
        print(
            f"fastnn_mean_ms mean={result['fastnn_mean_ms']['mean']:.3f} "
            f"min={result['fastnn_mean_ms']['min']:.3f} max={result['fastnn_mean_ms']['max']:.3f}"
        )
        print("kernel totals mean_ms")
        for key, row in result["aggregate_totals"].items():
            if row["count"] or row["mean_ms"]:
                print(f"  {key:22s} count={row['count']:3d} total_ms={row['mean_ms']:.3f}")
        print(f"write_const pct={result['write_const_percent_of_fastnn_mean']:.2f}%")
        print(f"non_conv pct={result['non_conv_percent_of_fastnn_mean']:.2f}%")
        if args.json is not None:
            print(f"json {args.json}")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))

    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
