#!/usr/bin/env python3
"""YOLO COCO val2017 mAP evaluation: PyTorch vs fastnn across dtypes.

Usage::

    python scripts/yolo_coco_map.py
    python scripts/yolo_coco_map.py --models yolo11n --dtypes f32,f8,f8r,f4
    python scripts/yolo_coco_map.py --max-images 100
    python scripts/yolo_coco_map.py --coco-dir /path/to/coco
"""
from __future__ import annotations

import argparse
import gc
import io
import json
import os
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def _get_rss_mb() -> float:
    """Current process RSS in MB (Linux /proc/self/status)."""
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024.0

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_DTYPES = ("f32", "u4", "u8", "f8", "f8r", "f4", "i4cb")
DEFAULT_COCO_DIR = "/home/petrouil/Projects/YOLO_Validation/coco"
MODEL_SPECS: dict[str, dict[str, Any]] = {
    "yolo11n": {"pt_name": "yolo11n.pt"},
    "yolo11l": {"pt_name": "yolo11l.pt"},
}

# ---------------------------------------------------------------------------
# COCO data loading
# ---------------------------------------------------------------------------

def _load_coco(coco_dir: str) -> tuple[Any, list[dict], dict[int, int]]:
    """Load COCO annotations → (coco, images, cat_map)."""
    from pycocotools.coco import COCO
    ann_file = Path(coco_dir) / "annotations" / "instances_val2017.json"
    coco = COCO(str(ann_file))
    images = []
    for img_id in sorted(coco.getImgIds()):
        img_info = coco.loadImgs(img_id)[0]
        images.append({"id": img_info["id"], "file_name": img_info["file_name"],
                        "height": img_info["height"], "width": img_info["width"]})
    cat_map = _coco80_to_coco91(coco)
    return coco, images, cat_map

def _coco80_to_coco91(coco) -> dict[int, int]:
    """Map model class indices (0-79) to COCO category IDs (1-90)."""
    coco80_indices = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38,
        39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
        56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75,
        76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
    ]
    return {i: cat_id for i, cat_id in enumerate(coco80_indices)}

# ---------------------------------------------------------------------------
# Preprocessing (shared ultralytics LetterBox)
# ---------------------------------------------------------------------------

def _letterbox(
    img: np.ndarray, new_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Letterbox resize with padding → (padded_img, gain, (pad_w, pad_h))."""
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, (int(round(dw)), int(round(dh)))

def _preprocess(img_path: str, imgsz: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Load image, letterbox, normalize to [0,1] float32 CHW."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img, gain, pad = _letterbox(img, (imgsz, imgsz))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    return img, gain, pad

# ---------------------------------------------------------------------------
# fastnn executor builder (ONNX load + constant folding + AotExecutor)
# ---------------------------------------------------------------------------

def _format_attr_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return str(value.item())
        return ",".join(str(int(v)) if float(v).is_integer() else str(float(v))
                        for v in value.flatten())
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)

def _dtype_to_quantize(dtype: str) -> int | str | None:
    if dtype == "f32":
        return None
    if dtype == "u4":
        return 4
    if dtype == "u8":
        return 8
    return dtype  # "f8", "f8r", "f4"

def _build_fastnn_executor(onnx_path: Path, dtype: str) -> tuple[Any, str, str]:
    """Load ONNX, constant-fold (Shape/Gather/Add/Sub/Mul/Div for DFL paths),
    build AotExecutor. Returns (executor, input_name, output_name)."""
    import onnx
    from onnx import numpy_helper
    import fastnn as fnn
    from fastnn.io.onnx import _extract_attrs

    model = onnx.load(str(onnx_path))
    initializer_names = {init.name for init in model.graph.initializer}
    input_names = [i.name for i in model.graph.input if i.name not in initializer_names]
    output_names = [o.name for o in model.graph.output]
    input_shapes: dict[str, list[int]] = {}
    for i in model.graph.input:
        if i.name in input_names:
            dims = []
            for d in i.type.tensor_type.shape.dim:
                dv = d.dim_value if d.HasField("dim_value") else 0
                dims.append(int(dv) if dv > 0 else 1)
            input_shapes[i.name] = dims

    # Load initializers
    params: dict[str, Any] = {}
    const_values: dict[str, np.ndarray] = {}
    for init in model.graph.initializer:
        arr_raw = numpy_helper.to_array(init)
        const_values[init.name] = arr_raw
        params[init.name] = fnn.tensor(arr_raw.astype(np.float32, copy=False), list(arr_raw.shape))

    # Shape inference (graceful on custom opsets)
    known_shapes: dict[str, list[int]] = {}
    try:
        shape_model = onnx.shape_inference.infer_shapes(model)
        for vi in (list(shape_model.graph.input) + list(shape_model.graph.value_info)
                   + list(shape_model.graph.output)):
            dims = []
            tt = vi.type.tensor_type
            if tt.HasField("shape"):
                for d in tt.shape.dim:
                    dv = d.dim_value if d.HasField("dim_value") else 0
                    dims.append(int(dv) if dv > 0 else -1)
                known_shapes[vi.name] = dims
    except Exception:
        known_shapes = {}

    # Build nodes with constant folding
    nodes: list[dict[str, Any]] = []
    for idx, node in enumerate(model.graph.node):
        name = node.name or f"{node.op_type}_{idx}"
        attrs = _extract_attrs(node)
        if node.op_type == "Constant" and node.output:
            value = attrs.get("value")
            if not isinstance(value, np.ndarray):
                for attr in node.attribute:
                    if attr.name == "value":
                        value = numpy_helper.to_array(onnx.helper.get_attribute_value(attr))
                        break
            if isinstance(value, np.ndarray):
                const_values[node.output[0]] = value
                arr = value.astype(np.float32, copy=False) if value.dtype.kind == "f" else value.astype(np.float32)
                params[node.output[0]] = fnn.tensor(arr, list(arr.shape))
        elif node.output:
            out_name = node.output[0]
            try:
                if node.op_type == "Shape" and node.input[0] in known_shapes:
                    const_values[out_name] = np.asarray(known_shapes[node.input[0]], dtype=np.int64)
                elif node.op_type == "Gather" and node.input[0] in const_values and node.input[1] in const_values:
                    axis = int(attrs.get("axis", 0))
                    const_values[out_name] = np.take(
                        const_values[node.input[0]],
                        const_values[node.input[1]].astype(np.int64), axis=axis)
                elif (node.op_type in {"Add", "Sub", "Mul", "Div"} and len(node.input) >= 2
                      and node.input[0] in const_values and node.input[1] in const_values):
                    a, b = const_values[node.input[0]], const_values[node.input[1]]
                    const_values[out_name] = {"Add": a + b, "Sub": a - b,
                                               "Mul": a * b, "Div": np.floor_divide(a, b)}[node.op_type]
                if out_name in const_values and out_name not in params:
                    arr_raw = np.asarray(const_values[out_name])
                    params[out_name] = fnn.tensor(arr_raw.astype(np.float32, copy=False), list(arr_raw.shape))
            except Exception:
                pass
        item: dict[str, Any] = {"name": name, "op_type": node.op_type,
                                 "inputs": ",".join(node.input), "outputs": ",".join(node.output)}
        if node.op_type == "Slice" and len(node.input) >= 4:
            starts, ends, axes = (const_values.get(node.input[1]),
                                  const_values.get(node.input[2]), const_values.get(node.input[3]))
            if starts is not None and ends is not None and axes is not None:
                item["starts"] = str(int(np.asarray(starts).reshape(-1)[0]))
                item["ends"] = str(int(np.asarray(ends).reshape(-1)[0]))
                item["axes"] = str(int(np.asarray(axes).reshape(-1)[0]))
        for k, v in attrs.items():
            if k != "value" and k not in item:
                item[k] = _format_attr_value(v)
        if node.op_type == "Resize" and len(node.input) >= 3:
            scales_name = node.input[2]
            if scales_name in params:
                scales = np.asarray(params[scales_name].numpy(), dtype=np.float32).reshape(-1)
                if scales.size >= 4:
                    item["scale_h"] = str(int(scales[2]))
                    item["scale_w"] = str(int(scales[3]))
        nodes.append(item)

    del model, const_values, known_shapes
    gc.collect()
    quantize = _dtype_to_quantize(dtype)
    executor = fnn.AotExecutor(nodes, params, input_names, output_names,
                               input_shapes=input_shapes, quantize=quantize)
    return executor, input_names[0], output_names[0]

# ---------------------------------------------------------------------------
# Memory measurement
# ---------------------------------------------------------------------------

def _measure_memory(executor: Any) -> dict[str, Any]:
    """Structural memory stats from executor.memory_stats() (no forward pass)."""
    raw = dict(executor.memory_stats())
    arena_bytes = int(raw.get("arena_size", 0))
    write_const = int(raw.get("write_const_bytes", 0))
    return {
        "arena_size_mb": round(arena_bytes / 1_048_576.0, 2),
        "weights_mb": round(write_const / 1_048_576.0, 2),
        "workspace_mb": round(max(arena_bytes - write_const, 0) / 1_048_576.0, 2),
        "arena_size_bytes": arena_bytes,
        "write_const_bytes": write_const,
        "instructions": int(raw.get("instructions", 0)),
        "call_kernel_count": int(raw.get("call_kernel_count", 0)),
    }

def _measure_pytorch_model_memory(model: Any) -> dict[str, float]:
    """PyTorch weight memory via parameter inspection."""
    total = trainable = 0
    for p in model.model.parameters():
        nbytes = p.numel() * p.element_size()
        total += nbytes
        if p.requires_grad:
            trainable += nbytes
    return {"weights_mb": round(total / 1_048_576.0, 2),
            "trainable_mb": round(trainable / 1_048_576.0, 2)}

# ---------------------------------------------------------------------------
# Speed benchmark
# ---------------------------------------------------------------------------

def _benchmark_inference(executor: Any, input_name: str, input_tensor: Any,
                         warmup: int = 5, iters: int = 10) -> dict[str, float]:
    """Benchmark executor.forward() only — tensor must be pre-allocated."""
    for _ in range(warmup):
        executor.forward({input_name: input_tensor})
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        executor.forward({input_name: input_tensor})
        times.append((time.perf_counter() - t0) * 1000.0)
    return {"mean_ms": statistics.mean(times), "median_ms": statistics.median(times),
            "min_ms": min(times), "max_ms": max(times)}

# ---------------------------------------------------------------------------
# Inference runners
# ---------------------------------------------------------------------------

def _run_pytorch(model: Any, x_tensor: Any) -> np.ndarray:
    """Run PyTorch YOLO inference. Takes pre-allocated torch.Tensor."""
    import torch
    torch.set_num_threads(1)
    model.model.eval()
    with torch.no_grad():
        y = model.model(x_tensor)
    if isinstance(y, (tuple, list)):
        y = y[0]
    if hasattr(y, "detach"):
        y = y.detach().cpu().numpy()
    return y

# ---------------------------------------------------------------------------
# Postprocessing (shared ultralytics NMS)
# ---------------------------------------------------------------------------

def _postprocess(preds_bcn: np.ndarray, conf_thres: float = 0.001,
                 iou_thres: float = 0.6) -> list[np.ndarray]:
    """Apply ultralytics NMS to BCN-format predictions."""
    import torch
    from ultralytics.utils.nms import non_max_suppression
    preds_t = torch.from_numpy(preds_bcn.astype(np.float32))
    results = non_max_suppression(preds_t, conf_thres=conf_thres, iou_thres=iou_thres,
                                  nc=0, agnostic=False, multi_label=True, labels=(), max_det=300)
    return [r.cpu().numpy() if hasattr(r, "cpu") else np.asarray(r) for r in results]

def _scale_to_original(dets: np.ndarray, img_shape: tuple[int, int],
                       gain: float, pad: tuple[int, int]) -> np.ndarray:
    """Scale detection boxes from model input coords to original image coords."""
    if len(dets) == 0:
        return dets
    from ultralytics.utils.ops import scale_boxes
    dets_copy = dets.copy()
    ratio_pad = ([gain, gain], pad)
    dets_copy[:, :4] = scale_boxes((640, 640), dets_copy[:, :4], img_shape, ratio_pad=ratio_pad)
    return dets_copy

# ---------------------------------------------------------------------------
# COCO evaluation
# ---------------------------------------------------------------------------

def _preds_to_coco_json(all_dets: list[tuple[int, np.ndarray]],
                        cat_map: dict[int, int], img_id_to_idx: dict[int, int]) -> list[dict]:
    """Convert detections to COCO JSON format for COCOeval."""
    coco_results = []
    for img_id, dets in all_dets:
        for det in dets:
            x1, y1, x2, y2, conf, cls_idx = det
            cls_idx = int(cls_idx)
            if cls_idx not in cat_map:
                continue
            coco_results.append({"image_id": img_id, "category_id": cat_map[cls_idx],
                                 "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                                 "score": float(conf)})
    return coco_results

def _evaluate_coco(coco, coco_results: list[dict], img_ids: list[int],
                   quiet: bool = False) -> dict[str, float]:
    from pycocotools.cocoeval import COCOeval
    old_stdout = sys.stdout
    if quiet:
        sys.stdout = io.StringIO()
    try:
        coco_dt = coco.loadRes(coco_results)
        coco_eval = COCOeval(coco, coco_dt, "bbox")
        coco_eval.params.imgIds = img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    finally:
        if quiet:
            sys.stdout = old_stdout
    return {"mAP@0.5": float(coco_eval.stats[1]), "mAP@0.5:0.95": float(coco_eval.stats[0]),
            "AP_small": float(coco_eval.stats[3]), "AP_medium": float(coco_eval.stats[4]),
            "AP_large": float(coco_eval.stats[5])}

# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _export_yolo_pt_to_onnx(pt_path: str, onnx_path: Path, imgsz: int) -> Any:
    from ultralytics import YOLO
    model = YOLO(pt_path)
    exported = model.export(format="onnx", imgsz=imgsz, opset=12,
                            simplify=False, dynamic=False, half=False, device="cpu", verbose=False)
    exported_path = Path(exported)
    if exported_path.resolve() != onnx_path.resolve():
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        onnx_path.write_bytes(exported_path.read_bytes())
    return model

# ---------------------------------------------------------------------------
# Main — 4 phases: Setup → Memory → Timing → Accuracy
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="YOLO COCO val2017 mAP: PyTorch vs fastnn across dtypes")
    ap.add_argument("--models", default="yolo11n", help="Comma-separated model names")
    ap.add_argument("--dtypes", default="f32,f8,f8r,f4", help=f"Dtypes ({','.join(VALID_DTYPES)})")
    ap.add_argument("--coco-dir", default=DEFAULT_COCO_DIR, help="COCO dataset root")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-images", type=int, default=0, help="Limit images (0 = all)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--conf-thres", type=float, default=0.001)
    ap.add_argument("--iou-thres", type=float, default=0.6)
    ap.add_argument("--json-output", type=Path, default=None)
    ap.add_argument("--memory-stats", action="store_true", default=True,
                     help="Collect structural memory stats (default: on)")
    ap.add_argument("--no-memory-stats", dest="memory_stats", action="store_false",
                     help="Disable structural memory stats")
    ap.add_argument("--rss-memory", action="store_true", default=False, help=argparse.SUPPRESS)
    ap.add_argument("--memory", action="store_true", default=False, help=argparse.SUPPRESS)
    ap.add_argument("--short", action="store_true", help="Minimal output: summary table only")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.memory:
        print("warning: --memory is deprecated, use --rss-memory for RSS tracking or --memory-stats for structural stats", file=sys.stderr)
        args.rss_memory = True

    model_names = [m.strip() for m in args.models.split(",")]
    dtypes = [d.strip() for d in args.dtypes.split(",")]
    for d in dtypes:
        if d not in VALID_DTYPES:
            print(f"error: unknown dtype '{d}' (valid: {','.join(VALID_DTYPES)})")
            return 1

    # Phase 0: Load COCO
    print(f"Loading COCO from {args.coco_dir} ...")
    coco, images, cat_map = _load_coco(args.coco_dir)
    if args.max_images > 0:
        images = images[: args.max_images]
    print(f"  {len(images)} images, {len(cat_map)} classes")
    img_id_to_idx = {img["id"]: idx for idx, img in enumerate(images)}
    all_results: dict[str, dict[str, Any]] = {}

    for model_name in model_names:
        print(f"\n{'=' * 70}\n  MODEL: {model_name}\n{'=' * 70}")
        cache_dir = Path(f"/tmp/fastnn-yolo-coco/{model_name}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = cache_dir / "model.onnx"
        pt_name = MODEL_SPECS[model_name]["pt_name"]
        if not onnx_path.exists():
            print(f"  exporting {pt_name} -> {onnx_path}")
            _export_yolo_pt_to_onnx(pt_name, onnx_path, args.imgsz)

        from ultralytics import YOLO
        pt_model = YOLO(pt_name)

        # --- PyTorch: Memory ---
        pt_mem = _measure_pytorch_model_memory(pt_model)
        print(f"\n  PyTorch weights={pt_mem['weights_mb']:.1f}MB trainable={pt_mem['trainable_mb']:.1f}MB")

        # --- PyTorch: Timing (pre-allocate, warmup, benchmark) ---
        import torch
        torch.set_num_threads(1)
        pt_model.model.eval()
        x_first, _, _ = _preprocess(
            str(Path(args.coco_dir) / "val2017" / images[0]["file_name"]), args.imgsz)
        pt_x = torch.from_numpy(x_first[np.newaxis])
        for _ in range(args.warmup):
            _run_pytorch(pt_model, pt_x)
        pytorch_times: list[float] = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            _run_pytorch(pt_model, pt_x)
            pytorch_times.append((time.perf_counter() - t0) * 1000.0)
        pt_speed = {"mean_ms": statistics.mean(pytorch_times), "median_ms": statistics.median(pytorch_times),
                     "min_ms": min(pytorch_times), "max_ms": max(pytorch_times)}
        print(f"  PyTorch mAP@0.5=... speed={pt_speed['mean_ms']:.1f}ms/img")

        # --- PyTorch: Accuracy (NOT timed) ---
        pytorch_dets: list[tuple[int, np.ndarray]] = []
        for img_info in images:
            x_np, gain, pad = _preprocess(
                str(Path(args.coco_dir) / "val2017" / img_info["file_name"]), args.imgsz)
            y_pt = _run_pytorch(pt_model, torch.from_numpy(x_np[np.newaxis]))
            dets_list = _postprocess(y_pt, args.conf_thres, args.iou_thres)
            dets = dets_list[0] if dets_list else np.empty((0, 6))
            dets = _scale_to_original(dets, (img_info["height"], img_info["width"]), gain, pad)
            pytorch_dets.append((img_info["id"], dets))
        pytorch_coco = _preds_to_coco_json(pytorch_dets, cat_map, img_id_to_idx)
        eval_ids = [img["id"] for img in images]
        pt_metrics = _evaluate_coco(coco, pytorch_coco, eval_ids, quiet=args.short)
        print(f"  PyTorch mAP@0.5={pt_metrics['mAP@0.5']:.4f}  "
              f"mAP@0.5:0.95={pt_metrics['mAP@0.5:0.95']:.4f}  "
              f"speed={pt_speed['mean_ms']:.1f}ms/img")

        model_results: dict[str, Any] = {"pytorch": {"metrics": pt_metrics, "speed": pt_speed,
            "memory_stats": pt_mem, "num_predictions": len(pytorch_coco)}, "fastnn": {}}
        del pt_model
        gc.collect()

        # --- fastnn phases per dtype ---
        import fastnn as fnn
        for dtype in dtypes:
            if not args.short:
                print(f"\n  --- dtype={dtype} ---")
            try:
                # Phase 1: Setup
                executor, input_name, output_name = _build_fastnn_executor(onnx_path, dtype)
                # Phase 2: Memory
                fastnn_mem = _measure_memory(executor) if args.memory_stats else {}
                # Phase 3: Timing (pre-allocate, warmup, benchmark)
                fx = fnn.tensor(x_first[np.newaxis], list(x_first[np.newaxis].shape))
                speed = _benchmark_inference(executor, input_name, fx, args.warmup, args.iters)
                # Phase 4: Accuracy (NOT timed)
                fastnn_dets: list[tuple[int, np.ndarray]] = []
                for img_info in images:
                    x_np, gain, pad = _preprocess(
                        str(Path(args.coco_dir) / "val2017" / img_info["file_name"]), args.imgsz)
                    fx_img = fnn.tensor(x_np[np.newaxis], list(x_np[np.newaxis].shape))
                    result = executor.forward({input_name: fx_img})
                    y_fn = result[output_name].numpy()
                    dets_list = _postprocess(y_fn, args.conf_thres, args.iou_thres)
                    dets = dets_list[0] if dets_list else np.empty((0, 6))
                    dets = _scale_to_original(dets, (img_info["height"], img_info["width"]), gain, pad)
                    fastnn_dets.append((img_info["id"], dets))
                coco_results = _preds_to_coco_json(fastnn_dets, cat_map, img_id_to_idx)
                metrics = _evaluate_coco(coco, coco_results, eval_ids, quiet=True)

                delta5 = metrics["mAP@0.5"] - pt_metrics["mAP@0.5"]
                delta595 = metrics["mAP@0.5:0.95"] - pt_metrics["mAP@0.5:0.95"]
                speedup = pt_speed["mean_ms"] / max(speed["mean_ms"], 0.001)

                print(f"  mAP@0.5={metrics['mAP@0.5']:.4f}  mAP@0.5:0.95={metrics['mAP@0.5:0.95']:.4f}  "
                      f"speed={speed['mean_ms']:.1f}ms/img  speedup={speedup:.2f}x")
                print(f"  ΔmAP@0.5={delta5:+.4f}  ΔmAP@0.5:0.95={delta595:+.4f}")
                if fastnn_mem:
                    print(f"  memory: arena={fastnn_mem['arena_size_mb']:.1f}MB  "
                          f"weights={fastnn_mem['weights_mb']:.1f}MB  "
                          f"workspace={fastnn_mem['workspace_mb']:.1f}MB")

                entry: dict[str, Any] = {"metrics": metrics, "speed": speed,
                    "delta_vs_pytorch": {"mAP@0.5": delta5, "mAP@0.5:0.95": delta595},
                    "speedup": speedup, "num_predictions": len(coco_results)}
                if fastnn_mem:
                    entry["memory_stats"] = fastnn_mem
                model_results["fastnn"][dtype] = entry
            except Exception as exc:
                import traceback
                print(f"  ERROR: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                model_results["fastnn"][dtype] = {"error": f"{type(exc).__name__}: {exc}"}

        all_results[model_name] = model_results

    # --- Summary table ---
    print(f"\n{'=' * 70}\n  SUMMARY\n{'=' * 70}")
    for model_name, mr in all_results.items():
        pt = mr["pytorch"]
        print(f"\n  {model_name}:")
        print(f"    PyTorch:  mAP@0.5={pt['metrics']['mAP@0.5']:.4f}  "
              f"mAP@0.5:0.95={pt['metrics']['mAP@0.5:0.95']:.4f}  "
              f"{pt['speed']['mean_ms']:.1f}ms/img  weights={pt['memory_stats']['weights_mb']:.1f}MB")
        for dtype, fr in mr["fastnn"].items():
            if "error" in fr:
                print(f"    {dtype:6s}: ERROR {fr['error']}")
            else:
                m, s, d, mem = fr["metrics"], fr["speed"], fr["delta_vs_pytorch"], fr.get("memory_stats", {})
                line = (f"    {dtype:6s}: mAP@0.5={m['mAP@0.5']:.4f}  "
                        f"mAP@0.5:0.95={m['mAP@0.5:0.95']:.4f}  "
                        f"{s['mean_ms']:.1f}ms/img ({fr['speedup']:.2f}x)  "
                        f"Δ={d['mAP@0.5']:+.4f}/{d['mAP@0.5:0.95']:+.4f}")
                if mem:
                    line += f"  arena={mem['arena_size_mb']:.1f}MB  weights={mem['weights_mb']:.1f}MB"
                print(line)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(all_results, indent=2, sort_keys=True))
        print(f"\nResults written to {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
