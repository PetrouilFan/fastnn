#!/usr/bin/env python3
"""Compare YOLO CPU runtime thread settings across fastnn, PyTorch, and Ultralytics.

This is roadmap Task A2 from docs/plans/yolo-performance-roadmap.md.
Each configuration runs in a fresh subprocess so thread environment variables
are applied before importing native libraries.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any


FASTNN_CASE = r'''
import json
import os
import subprocess
import sys
from pathlib import Path

threads = int(os.environ["FASTNN_MATRIX_THREADS"])
profile_json = Path(os.environ["FASTNN_MATRIX_JSON"])
cmd = [
    sys.executable,
    "scripts/yolo_compare_fastnn_pytorch.py",
    "--profile",
    "--profile-top",
    "20",
    "--warmup",
    os.environ["FASTNN_MATRIX_WARMUP"],
    "--iters",
    os.environ["FASTNN_MATRIX_ITERS"],
    "--profile-json",
    str(profile_json),
]
env = os.environ.copy()
env["OPENBLAS_NUM_THREADS"] = str(threads)
env.pop("FASTNN_DISABLE_OPENBLAS_CONV_GEMM", None)
proc = subprocess.run(cmd, cwd=os.environ["FASTNN_MATRIX_CWD"], env=env, text=True, capture_output=True)
if proc.returncode != 0:
    print(json.dumps({"ok": False, "returncode": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:]}))
    raise SystemExit(0)
data = json.loads(profile_json.read_text())
print(json.dumps({
    "ok": True,
    "backend": "fastnn",
    "threads": threads,
    "speed": data.get("fastnn_speed"),
    "vs_pytorch": data.get("fastnn_vs_pytorch"),
    "profile_top": data.get("fastnn_profile_top", [])[:8],
}))
'''

TORCH_CASE = r'''
import json, os, statistics, time
import numpy as np
import torch
from ultralytics import YOLO

threads = int(os.environ["FASTNN_MATRIX_THREADS"])
torch.set_num_threads(threads)
imgsz = int(os.environ["FASTNN_MATRIX_IMGSZ"])
warmup = int(os.environ["FASTNN_MATRIX_WARMUP"])
iters = int(os.environ["FASTNN_MATRIX_ITERS"])
model = YOLO(os.environ["FASTNN_MATRIX_PT"])
try:
    model.fuse()
except Exception:
    pass
net = model.model.eval().cpu()
x = torch.from_numpy(np.random.default_rng(0).random((1, 3, imgsz, imgsz), dtype=np.float32))
with torch.no_grad():
    for _ in range(warmup):
        y = net(x)
        y0 = y[0] if isinstance(y, (tuple, list)) else y
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        y = net(x)
        y0 = y[0] if isinstance(y, (tuple, list)) else y
        times.append((time.perf_counter() - t0) * 1000.0)
print(json.dumps({
    "ok": True,
    "backend": os.environ["FASTNN_MATRIX_BACKEND"],
    "threads": threads,
    "shape": list(y0.shape),
    "speed": {
        "iters": iters,
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
    },
    "torch_num_threads": torch.get_num_threads(),
}))
'''


def _parse_threads(value: str) -> list[int]:
    out = []
    for part in value.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise ValueError("thread list is empty")
    return out


def _run_python(code: str, env: dict[str, str], cwd: Path) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        return {
            "ok": False,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
        }
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return {"ok": False, "returncode": 0, "stdout": proc.stdout, "stderr": proc.stderr, "error": "empty stdout"}
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
            "error": f"json decode: {exc}",
        }


def _best(rows: list[dict[str, Any]], backend: str) -> dict[str, Any] | None:
    candidates = [r for r in rows if r.get("ok") and r.get("backend") == backend and r.get("speed")]
    if not candidates:
        return None
    return min(candidates, key=lambda r: float(r["speed"]["mean_ms"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="1,2,4,8", help="Comma-separated thread settings")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--pt", default="yolov8n.pt")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--skip-fastnn", action="store_true")
    ap.add_argument("--skip-torch", action="store_true")
    ap.add_argument("--skip-ultralytics", action="store_true")
    args = ap.parse_args()

    cwd = Path.cwd()
    threads = _parse_threads(args.threads)
    tmp_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "fastnn-yolo-runtime-matrix"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    base_env = os.environ.copy()
    base_env.update({
        "FASTNN_MATRIX_CWD": str(cwd),
        "FASTNN_MATRIX_WARMUP": str(args.warmup),
        "FASTNN_MATRIX_ITERS": str(args.iters),
        "FASTNN_MATRIX_IMGSZ": str(args.imgsz),
        "FASTNN_MATRIX_PT": args.pt,
    })

    for t in threads:
        if not args.skip_fastnn:
            env = base_env.copy()
            env["FASTNN_MATRIX_THREADS"] = str(t)
            env["FASTNN_MATRIX_JSON"] = str(tmp_dir / f"fastnn_threads_{t}.json")
            row = _run_python(FASTNN_CASE, env, cwd)
            rows.append(row)
            if row.get("ok"):
                print(f"fastnn threads={t} mean_ms={row['speed']['mean_ms']:.3f}")
            else:
                print(f"fastnn threads={t} ERROR {row.get('error') or row.get('returncode')}")

        if not args.skip_torch:
            env = base_env.copy()
            env["FASTNN_MATRIX_THREADS"] = str(t)
            env["FASTNN_MATRIX_BACKEND"] = "pytorch_raw"
            row = _run_python(TORCH_CASE, env, cwd)
            rows.append(row)
            if row.get("ok"):
                print(f"pytorch_raw threads={t} mean_ms={row['speed']['mean_ms']:.3f}")
            else:
                print(f"pytorch_raw threads={t} ERROR {row.get('error') or row.get('returncode')}")

        if not args.skip_ultralytics:
            env = base_env.copy()
            env["FASTNN_MATRIX_THREADS"] = str(t)
            env["FASTNN_MATRIX_BACKEND"] = "ultralytics_raw"
            row = _run_python(TORCH_CASE, env, cwd)
            rows.append(row)
            if row.get("ok"):
                print(f"ultralytics_raw threads={t} mean_ms={row['speed']['mean_ms']:.3f}")
            else:
                print(f"ultralytics_raw threads={t} ERROR {row.get('error') or row.get('returncode')}")

    best = {name: _best(rows, name) for name in ["fastnn", "pytorch_raw", "ultralytics_raw"]}
    summary = {"threads": threads, "warmup": args.warmup, "iters": args.iters, "rows": rows, "best": best}

    print("best")
    for name, row in best.items():
        if row:
            print(f"  {name}: threads={row['threads']} mean_ms={row['speed']['mean_ms']:.3f} median_ms={row['speed']['median_ms']:.3f}")
        else:
            print(f"  {name}: unavailable")

    best_fastnn = best.get("fastnn")
    if best_fastnn is not None:
        fast = float(best_fastnn["speed"]["mean_ms"])
        ratios = {}
        for name in ["pytorch_raw", "ultralytics_raw"]:
            best_other = best.get(name)
            if best_other is not None:
                other = float(best_other["speed"]["mean_ms"])
                ratios[f"fastnn_vs_{name}"] = fast / other
                print(f"  fastnn/{name} ratio={fast / other:.3f}")
        summary["ratios"] = ratios

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(f"json {args.json}")

    return 0 if all(r.get("ok") for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
