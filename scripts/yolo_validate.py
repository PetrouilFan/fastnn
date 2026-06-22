#!/usr/bin/env python3
"""YOLO validation: FP32 vs U8 accuracy comparison using Ultralytics val."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# Headless: suppress any GUI backend before any import
os.environ.setdefault("MPLBACKEND", "Agg")

_COCO_ROOT = Path("/home/petrouil/data/coco")


def _check_coco() -> bool:
    exists = _COCO_ROOT.is_dir()
    if not exists:
        print(f"COCO data not found at {_COCO_ROOT}")
        print(
            "SKIP: validation requires COCO val2017 images and annotations "
            "at /home/petrouil/data/coco"
        )
    else:
        print(f"COCO data found at {_COCO_ROOT}")
    return exists


def _run_yolo_val(
    model_path: str,
    data: str,
    imgsz: int,
    device: str,
    project: str,
    name: str,
    label: str,
) -> dict[str, Any] | None:
    cmd = [
        "yolo", "val",
        "model", model_path,
        "data", data,
        "imgsz", str(imgsz),
        "device", device,
        "save_json",
        "project", project,
        "name", name,
        "exist_ok",
    ]
    print(f"[{label}] Running: {' '.join(cmd)}")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    except subprocess.TimeoutExpired:
        print(f"[{label}] Timed out after 1200s")
        return None
    if r.returncode != 0:
        print(f"[{label}] returncode={r.returncode}")
        stderr_tail = r.stderr[-1500:] if len(r.stderr) > 1500 else r.stderr
        print(f"[{label}] stderr: {stderr_tail}")
        return None
    stdout = r.stdout
    if len(stdout) > 3000:
        print(f"[{label}] stdout: (trimmed) ...{stdout[-3000:]}")
    else:
        print(f"[{label}] stdout: {stdout}")
    if r.stderr:
        print(f"[{label}] stderr: {r.stderr[-500:]}")

    out_dir = Path(project) / name
    candidates = sorted(out_dir.rglob("*.json"))
    best = None
    for p in candidates:
        lower = p.name.lower()
        if lower in ("results.json", "val.json"):
            best = p
            break
    if best is None:
        for p in candidates:
            if "result" in p.name.lower():
                best = p
                break
    if best is None and candidates:
        best = candidates[0]

    if best and best.exists():
        try:
            return json.loads(best.read_text())
        except Exception as exc:
            print(f"[{label}] JSON parse error from {best}: {exc}")
    else:
        print(f"[{label}] no results JSON found under {out_dir}")
    return None


def _export_onnx(pt_path: str, onnx_path: Path, imgsz: int, int8: bool = False) -> None:
    from ultralytics import YOLO
    model = YOLO(pt_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=12,
        simplify=False,
        dynamic=False,
        half=False,
        int8=int8,
        device="cpu",
        verbose=False,
    )
    exported_path = Path(exported)
    if exported_path.resolve() != onnx_path.resolve():
        shutil.copy2(str(exported_path), str(onnx_path))
    print(f"  exported -> {onnx_path}")


def _summarize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    box = metrics.get("box", {})
    return {
        "map50-95": box.get("map", metrics.get("metrics/mAP50-95")),
        "map50": box.get("map50", metrics.get("metrics/mAP50")),
        "precision": box.get("precision", metrics.get("metrics/precision")),
        "recall": box.get("recall", metrics.get("metrics/recall")),
    }


def _delta_metrics(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for key in ("map50-95", "map50", "precision", "recall"):
        va = a.get(key)
        vb = b.get(key)
        if va is not None and vb is not None and isinstance(va, (int, float)):
            deltas[key] = round(vb - va, 6)
    return deltas


def _print_summary(tag: str, metrics: dict[str, Any] | None) -> None:
    if metrics is None:
        print(f"  {tag:4s}: (no data)")
        return
    parts = [f"  {tag:4s}:"]
    for k in ("map50-95", "map50", "precision", "recall"):
        v = metrics.get(k)
        if v is not None:
            parts.append(f"{k}={v:.4f}")
    print("  ".join(parts))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="YOLO FP32 vs U8 accuracy validation on COCO"
    )
    ap.add_argument("--model", default="yolov8n.pt",
                    help="Ultralytics YOLO .pt model path")
    ap.add_argument("--data", default="coco.yaml",
                    help="Dataset YAML or path (default: coco.yaml → uses COCO at /home/petrouil/data/coco)")
    ap.add_argument("--imgsz", type=int, default=320,
                    help="Inference image size (pixels)")
    ap.add_argument("--device", default="cpu",
                    help="Device: cpu, cuda:0, mps, ...")
    ap.add_argument("--output", type=Path,
                    default=Path("/tmp/yolo_val_results.json"),
                    help="Path for comparison results JSON")

    args = ap.parse_args()

    # --- COCO availability (soft check) ---
    if args.data == "coco.yaml" and not _check_coco():
        return 0

    model_pt = args.model
    imgsz = args.imgsz
    device = args.device
    data = args.data

    base = Path("/tmp/yolo_val_results")
    base.mkdir(parents=True, exist_ok=True)

    # --- FP32 -----------------------------------------------------------------
    print("\n=== FP32 validation ===")
    fp32_onnx = base / "models" / f"{Path(model_pt).stem}_fp32.onnx"
    if not fp32_onnx.exists():
        _export_onnx(model_pt, fp32_onnx, imgsz, int8=False)
    fp32_metrics = _run_yolo_val(
        str(fp32_onnx), data, imgsz, device,
        str(base / "val"), "fp32", "FP32",
    )

    # --- U8 (INT8) ------------------------------------------------------------
    print("\n=== U8 (INT8) validation ===")
    u8_onnx = base / "models" / f"{Path(model_pt).stem}_int8.onnx"
    if not u8_onnx.exists():
        _export_onnx(model_pt, u8_onnx, imgsz, int8=True)
    u8_metrics = _run_yolo_val(
        str(u8_onnx), data, imgsz, device,
        str(base / "val"), "u8", "U8",
    )

    # --- Build report ---------------------------------------------------------
    fp32_summary = _summarize_metrics(fp32_metrics) if fp32_metrics else None
    u8_summary = _summarize_metrics(u8_metrics) if u8_metrics else None

    results: dict[str, Any] = {
        "model": model_pt,
        "imgsz": imgsz,
        "device": device,
        "data": data,
        "fp32": fp32_summary,
        "u8": u8_summary,
    }
    if fp32_summary and u8_summary:
        results["delta_u8_minus_fp32"] = _delta_metrics(fp32_summary, u8_summary)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"\nResults written to {args.output}")

    # --- Console summary ------------------------------------------------------
    print()
    _print_summary("FP32", fp32_summary)
    _print_summary("U8", u8_summary)
    if "delta_u8_minus_fp32" in results:
        d = results["delta_u8_minus_fp32"]
        parts = ["  DELTA:"]
        for k in ("map50-95", "map50", "precision", "recall"):
            if k in d:
                parts.append(f"{k}={d[k]:+.4f}")
        print("  ".join(parts))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
