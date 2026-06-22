#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/petrouil/Projects/github/fastnn")
SCRIPT = ROOT / "scripts" / "yolo_compare_fastnn_pytorch.py"


def run(args: list[str]) -> str:
    p = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if p.returncode != 0:
        raise RuntimeError(p.stderr)
    return p.stdout


def extract(line_prefix: str, text: str) -> dict:
    for raw in text.splitlines():
        if raw.startswith(line_prefix):
            payload = raw.split(" ", 1)[1]
            return json.loads(payload)
    raise KeyError(line_prefix)


def main() -> int:
    cases = {
        "fp32": [],
        "u8": [],
        "u4": [],
    }
    for bit in ["fp32", "u8", "u4"]:
        print(f"run {bit}")
        args = [
            "--pt", "yolov8n.pt",
            "--imgsz", "320",
            "--warmup", "1",
            "--iters", "2",
        ]
        if bit != "fp32":
            args += ["--quantize", bit[1]]
        args += ["--profile-json", f"/tmp/yolo_{bit}_bench.json"]
        out = run(args)
        cases[bit] = {
            "pytorch_ms": extract("pytorch", out)["speed"]["min_ms"],
            "fastnn_ms": extract("fastnn", out)["speed"]["min_ms"],
            "max_abs": extract("fastnn", out)["vs_pytorch"]["max_abs"],
        }
    for k, v in cases.items():
        print(f"{k}: fastnn={v['fastnn_ms']:.3f} ms (pytorch={v['pytorch_ms']:.3f} ms) max_abs={v['max_abs']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
