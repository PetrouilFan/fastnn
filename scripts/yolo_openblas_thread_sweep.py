#!/usr/bin/env python3
"""Sweep OPENBLAS_NUM_THREADS for fastnn YOLO inference.

This script assumes the current Python environment already has fastnn installed
with the desired features. For OpenBLAS testing, build first with:

    VIRTUAL_ENV=/home/petrouil/Projects/github/fastnn/.venv \
      .venv/bin/python -m maturin develop --release --features 'prepared-plan openblas'

It launches each measurement in a fresh subprocess so OPENBLAS_NUM_THREADS is
set before importing fastnn/OpenBLAS.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]


def _parse_fastnn_json(stdout: str) -> dict[str, Any]:
    for line in stdout.splitlines():
        if line.startswith("fastnn "):
            return json.loads(line[len("fastnn ") :])
    raise RuntimeError(f"no 'fastnn ' JSON line found in output:\n{stdout[-2000:]}")


def run_case(
    threads: int,
    onnx: Path,
    warmup: int,
    iters: int,
    profile_top: int,
    python: str,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = str(threads)
    env.setdefault("OMP_NUM_THREADS", "1")
    cmd = [
        python,
        "scripts/yolo_compare_fastnn_pytorch.py",
        "--onnx",
        str(onnx),
        "--profile",
        "--profile-top",
        str(profile_top),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "threads": threads,
            "ok": False,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }
    fastnn = _parse_fastnn_json(proc.stdout)
    profile_top_items = []
    for line in proc.stdout.splitlines():
        if line.startswith("fastnn_profile_top "):
            profile_top_items = json.loads(line[len("fastnn_profile_top ") :])
            break
    return {
        "threads": threads,
        "ok": True,
        "speed": fastnn.get("speed", {}),
        "vs_pytorch": fastnn.get("vs_pytorch", {}),
        "profile_top": profile_top_items,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", type=Path, default=Path("yolov8n.onnx"))
    ap.add_argument("--threads", default="1,2,4,8", help="comma-separated OpenBLAS thread counts")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--profile-top", type=int, default=8)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    threads = [int(x.strip()) for x in args.threads.split(",") if x.strip()]
    results = [
        run_case(t, args.onnx, args.warmup, args.iters, args.profile_top, args.python)
        for t in threads
    ]
    ok = [r for r in results if r.get("ok")]
    best = min(ok, key=lambda r: r["speed"].get("mean_ms", float("inf"))) if ok else None
    payload = {
        "onnx": str(args.onnx),
        "warmup": args.warmup,
        "iters": args.iters,
        "threads": threads,
        "results": results,
        "best": best,
    }

    print("OpenBLAS thread sweep")
    print(f"onnx={args.onnx} warmup={args.warmup} iters={args.iters}")
    print(f"{'threads':>8} {'mean_ms':>10} {'median_ms':>10} {'min_ms':>10} {'max_ms':>10} {'conv_silu_ms':>14} {'max_abs':>12}")
    for r in results:
        if not r.get("ok"):
            print(f"{r['threads']:>8} FAILED rc={r.get('returncode')}")
            continue
        speed = r["speed"]
        conv_silu_total = None
        for item in r.get("profile_top", []):
            if item.get("kernel_name") == "conv2d_silu":
                conv_silu_total = item.get("total_ms")
                break
        print(
            f"{r['threads']:>8} "
            f"{speed.get('mean_ms', float('nan')):>10.3f} "
            f"{speed.get('median_ms', float('nan')):>10.3f} "
            f"{speed.get('min_ms', float('nan')):>10.3f} "
            f"{speed.get('max_ms', float('nan')):>10.3f} "
            f"{(conv_silu_total if conv_silu_total is not None else float('nan')):>14.3f} "
            f"{r.get('vs_pytorch', {}).get('max_abs', float('nan')):>12.3e}"
        )
    if best:
        print(
            "best "
            f"threads={best['threads']} "
            f"mean_ms={best['speed'].get('mean_ms'):.3f} "
            f"median_ms={best['speed'].get('median_ms'):.3f}"
        )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"wrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
