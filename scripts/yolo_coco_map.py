#!/usr/bin/env python3
"""YOLO COCO val2017 mAP evaluation: PyTorch vs fastnn across dtypes.

Runs inference on COCO val2017 images with both PyTorch and fastnn
(AOT executor with optional quantization), applies shared ultralytics
NMS + scale_boxes postprocessing, and evaluates with pycocotools COCOeval.

Usage::

    # Default: yolo11n, all dtypes
    python scripts/yolo_coco_map.py

    # Specific models + dtypes
    python scripts/yolo_coco_map.py \\
        --models yolo11n --dtypes f32,f8,f8r,f4

    # Limit number of images
    python scripts/yolo_coco_map.py --max-images 100

    # Custom COCO path
    python scripts/yolo_coco_map.py --coco-dir /path/to/coco
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import gc
import io
import os
import resource
import sys

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Memory measurement
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
    # Fallback: resource.getrusage (returns KB on Linux)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024.0


class _MemoryTracker:
    """Track RSS memory during an inference loop.

    Usage::

        tracker = _MemoryTracker(label="PyTorch")
        # ... model loaded, ready to infer ...
        tracker.baseline()            # snapshot baseline RSS
        for img in images:
            inference(img)
            tracker.sample()          # track per-image RSS
        stats = tracker.summary()     # compute peak, delta, per-image
    """

    def __init__(self, enabled: bool = True, label: str = ""):
        self.enabled = enabled
        self.label = label
        self._baseline_mb: float = 0.0
        self._peak_mb: float = 0.0
        self._samples: list[float] = []

    def baseline(self) -> None:
        """Record baseline RSS (call before inference loop)."""
        if not self.enabled:
            return
        gc.collect()
        self._baseline_mb = _get_rss_mb()
        self._peak_mb = self._baseline_mb
        self._samples = []

    def sample(self) -> None:
        """Record current RSS (call once per iteration)."""
        if not self.enabled:
            return
        rss = _get_rss_mb()
        self._samples.append(rss)
        if rss > self._peak_mb:
            self._peak_mb = rss

    def summary(self) -> dict[str, float]:
        """Return memory stats: baseline, peak, delta, per_image_avg."""
        if not self.enabled:
            return {}
        delta = self._peak_mb - self._baseline_mb
        per_image = delta / max(len(self._samples), 1)
        return {
            "baseline_rss_mb": round(self._baseline_mb, 2),
            "peak_rss_mb": round(self._peak_mb, 2),
            "delta_rss_mb": round(delta, 2),
            "per_image_mb": round(per_image, 3),
            "num_samples": len(self._samples),
        }


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_DTYPES = ("f32", "u4", "u8", "f8", "f8r", "f4")

DEFAULT_COCO_DIR = "/home/petrouil/Projects/YOLO_Validation/coco"

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "yolo11n": {"pt_name": "yolo11n.pt"},
    "yolo11l": {"pt_name": "yolo11l.pt"},
}


# ---------------------------------------------------------------------------
# COCO data loading
# ---------------------------------------------------------------------------


def _load_coco(coco_dir: str) -> tuple[Any, list[dict], dict[int, int]]:
    """Load COCO annotations and build image list + class mapping.

    Returns:
        coco: pycocotools COCO instance
        images: list of {id, file_name, height, width}
        cat_map: {model_class_idx: coco_category_id}  (coco80 → coco91)
    """
    from pycocotools.coco import COCO

    ann_file = Path(coco_dir) / "annotations" / "instances_val2017.json"
    coco = COCO(str(ann_file))

    images = []
    for img_id in sorted(coco.getImgIds()):
        img_info = coco.loadImgs(img_id)[0]
        images.append({
            "id": img_info["id"],
            "file_name": img_info["file_name"],
            "height": img_info["height"],
            "width": img_info["width"],
        })

    # coco80 → coco91 mapping (standard ultralytics ordering)
    cat_map = _coco80_to_coco91(coco)

    return coco, images, cat_map


def _coco80_to_coco91(coco) -> dict[int, int]:
    """Map model class indices (0-79) to COCO category IDs (1-90)."""
    # Standard mapping used by ultralytics
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
    img: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Letterbox resize with padding. Returns (padded_img, gain, (pad_w, pad_h))."""
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
    """Load image, letterbox, normalize to [0,1] float32 CHW.

    Returns: (tensor[3,640,640], gain, (pad_w, pad_h))
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img, gain, pad = _letterbox(img, (imgsz, imgsz))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    return img, gain, pad


# ---------------------------------------------------------------------------
# Inference runners
# ---------------------------------------------------------------------------


def _run_pytorch(model: Any, x_np: np.ndarray) -> np.ndarray:
    """Run PyTorch YOLO inference. Returns [1, 84, 8400] numpy array."""
    import torch

    torch.set_num_threads(1)
    model.model.eval()
    xt = torch.from_numpy(x_np)
    with torch.no_grad():
        y = model.model(xt)
    if isinstance(y, (tuple, list)):
        y = y[0]
    if hasattr(y, "detach"):
        y = y.detach().cpu().numpy()
    return y


def _dtype_to_quantize(dtype: str) -> int | str | None:
    if dtype == "f32":
        return None
    if dtype == "u4":
        return 4
    if dtype == "u8":
        return 8
    return dtype  # "f8", "f8r", "f4"


def _make_fastnn_executor(onnx_path: Path, dtype: str):
    """Build fastnn AotExecutor from ONNX file."""
    import onnx
    from onnx import numpy_helper

    import fastnn as fnn
    from fastnn.io.onnx import _extract_attrs

    model = onnx.load(str(onnx_path))
    initializer_names = {init.name for init in model.graph.initializer}
    input_names = [i.name for i in model.graph.input if i.name not in initializer_names]
    output_names = [o.name for o in model.graph.output]

    input_shapes = {}
    for i in model.graph.input:
        if i.name in input_names:
            dims = []
            for d in i.type.tensor_type.shape.dim:
                dv = d.dim_value if d.HasField("dim_value") else 0
                dims.append(int(dv) if dv > 0 else 1)
            input_shapes[i.name] = dims

    params = {}
    const_values: dict[str, np.ndarray] = {}
    for init in model.graph.initializer:
        arr_raw = numpy_helper.to_array(init)
        const_values[init.name] = arr_raw
        arr = arr_raw.astype(np.float32, copy=False)
        params[init.name] = fnn.tensor(arr, list(arr.shape))

    known_shapes: dict[str, list[int]] = {}
    try:
        shape_model = onnx.shape_inference.infer_shapes(model)
        for vi in list(shape_model.graph.input) + list(shape_model.graph.value_info) + list(shape_model.graph.output):
            dims = []
            tt = vi.type.tensor_type
            if tt.HasField("shape"):
                for d in tt.shape.dim:
                    dv = d.dim_value if d.HasField("dim_value") else 0
                    dims.append(int(dv) if dv > 0 else -1)
                known_shapes[vi.name] = dims
    except Exception:
        pass

    def _format_attr_value(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return str(value.item())
            return ",".join(
                str(int(v)) if float(v).is_integer() else str(float(v))
                for v in value.flatten()
            )
        if isinstance(value, (list, tuple)):
            return ",".join(str(v) for v in value)
        return str(value)

    nodes = []
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
                        const_values[node.input[1]].astype(np.int64),
                        axis=axis,
                    )
                elif node.op_type in {"Add", "Sub", "Mul", "Div"} and len(node.input) >= 2:
                    if node.input[0] in const_values and node.input[1] in const_values:
                        a = const_values[node.input[0]]
                        b = const_values[node.input[1]]
                        if node.op_type == "Add":
                            const_values[out_name] = a + b
                        elif node.op_type == "Sub":
                            const_values[out_name] = a - b
                        elif node.op_type == "Mul":
                            const_values[out_name] = a * b
                        elif node.op_type == "Div":
                            const_values[out_name] = np.floor_divide(a, b)
                if out_name in const_values and out_name not in params:
                    arr_raw = np.asarray(const_values[out_name])
                    arr = arr_raw.astype(np.float32, copy=False)
                    params[out_name] = fnn.tensor(arr, list(arr.shape))
            except Exception:
                pass
        item: dict[str, Any] = {
            "name": name,
            "op_type": node.op_type,
            "inputs": ",".join(node.input),
            "outputs": ",".join(node.output),
        }
        if node.op_type == "Slice" and len(node.input) >= 4:
            starts = const_values.get(node.input[1])
            ends = const_values.get(node.input[2])
            axes = const_values.get(node.input[3])
            if starts is not None and ends is not None and axes is not None:
                item["starts"] = str(int(np.asarray(starts).reshape(-1)[0]))
                item["ends"] = str(int(np.asarray(ends).reshape(-1)[0]))
                axes_arr = np.asarray(axes).reshape(-1)
                item["axes"] = str(int(axes_arr[0]))
        for k, v in attrs.items():
            if k == "value" or k in item:
                continue
            item[k] = _format_attr_value(v)
        if node.op_type == "Resize" and len(node.input) >= 3:
            scales_name = node.input[2]
            if scales_name in params:
                scales = np.asarray(params[scales_name].numpy(), dtype=np.float32).reshape(-1)
                if scales.size >= 4:
                    item["scale_h"] = str(int(scales[2]))
                    item["scale_w"] = str(int(scales[3]))
        nodes.append(item)

    quantize = _dtype_to_quantize(dtype)
    executor = fnn.AotExecutor(
        nodes, params, input_names, output_names,
        input_shapes=input_shapes, quantize=quantize,
    )
    return executor, input_names[0], output_names[0]


def _run_fastnn(onnx_path: Path, dtype: str, x_np: np.ndarray) -> np.ndarray:
    """Run fastnn AOT executor. Returns [1, 84, 8400] numpy array."""
    import fastnn as fnn

    executor, input_name, output_name = _make_fastnn_executor(onnx_path, dtype)
    fx = fnn.tensor(x_np, list(x_np.shape))
    result = executor.forward({input_name: fx})
    y = result[output_name].numpy()
    return y


# ---------------------------------------------------------------------------
# Postprocessing (shared ultralytics NMS)
# ---------------------------------------------------------------------------


def _postprocess(
    preds_bcn: np.ndarray,
    conf_thres: float = 0.001,
    iou_thres: float = 0.6,
) -> list[np.ndarray]:
    """Apply ultralytics NMS to BCN-format predictions.

    Args:
        preds_bcn: [batch, 84, 8400] raw model output
        conf_thres: confidence threshold
        iou_thres: NMS IoU threshold

    Returns:
        List of [N, 6] arrays per image with [x1, y1, x2, y2, conf, cls]
    """
    import torch
    from ultralytics.utils.nms import non_max_suppression

    # Convert to torch tensor in BCN format [batch, 84, 8400]
    preds_t = torch.from_numpy(preds_bcn.astype(np.float32))
    results = non_max_suppression(
        preds_t,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        nc=0,  # auto-detect classes from channel dim
        agnostic=False,
        multi_label=True,
        labels=(),
        max_det=300,
    )
    # results is list[Tensor] with [N, 6] = [x1,y1,x2,y2,conf,cls]
    return [r.cpu().numpy() if hasattr(r, "cpu") else np.asarray(r) for r in results]


def _scale_to_original(
    dets: np.ndarray,
    img_shape: tuple[int, int],
    gain: float,
    pad: tuple[int, int],
) -> np.ndarray:
    """Scale detection boxes from model input coords to original image coords."""
    if len(dets) == 0:
        return dets
    from ultralytics.utils.ops import scale_boxes

    # scale_boxes expects ratio_pad=(ratio, (pad_h, pad_w))
    # where ratio is a tuple/list of (ratio_h, ratio_w)
    model_shape = (640, 640)
    dets_copy = dets.copy()
    boxes = dets_copy[:, :4]
    ratio_pad = ([gain, gain], pad)
    scaled = scale_boxes(model_shape, boxes, img_shape, ratio_pad=ratio_pad)
    dets_copy[:, :4] = scaled
    return dets_copy


# ---------------------------------------------------------------------------
# COCO evaluation
# ---------------------------------------------------------------------------


def _preds_to_coco_json(
    all_dets: list[tuple[int, np.ndarray]],
    cat_map: dict[int, int],
    img_id_to_idx: dict[int, int],
) -> list[dict]:
    """Convert detections to COCO JSON format for COCOeval.

    Args:
        all_dets: list of (image_id, detections[N,6]) tuples
        cat_map: model class idx → COCO category ID
        img_id_to_idx: image_id → dataset index

    Returns:
        List of COCO-format annotation dicts
    """
    coco_results = []
    for img_id, dets in all_dets:
        if len(dets) == 0:
            continue
        for det in dets:
            x1, y1, x2, y2, conf, cls_idx = det
            cls_idx = int(cls_idx)
            if cls_idx not in cat_map:
                continue
            cat_id = cat_map[cls_idx]
            w = x2 - x1
            h = y2 - y1
            coco_results.append({
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(conf),
            })
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

    return {
        "mAP@0.5": float(coco_eval.stats[1]),
        "mAP@0.5:0.95": float(coco_eval.stats[0]),
        "AP_small": float(coco_eval.stats[3]),
        "AP_medium": float(coco_eval.stats[4]),
        "AP_large": float(coco_eval.stats[5]),
    }


# ---------------------------------------------------------------------------
# Speed benchmark
# ---------------------------------------------------------------------------


def _benchmark_inference(fn, warmup: int = 5, iters: int = 10) -> dict[str, float]:
    """Benchmark a zero-arg inference callable."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def _export_yolo_pt_to_onnx(pt_path: str, onnx_path: Path, imgsz: int) -> Any:
    from ultralytics import YOLO

    model = YOLO(pt_path)
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=12,
        simplify=False,
        dynamic=False,
        half=False,
        device="cpu",
        verbose=False,
    )
    exported_path = Path(exported)
    if exported_path.resolve() != onnx_path.resolve():
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        onnx_path.write_bytes(exported_path.read_bytes())
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="YOLO COCO val2017 mAP: PyTorch vs fastnn across dtypes",
    )
    ap.add_argument("--models", default="yolo11n",
                     help="Comma-separated model names (default: yolo11n)")
    ap.add_argument("--dtypes", default="f32,f8,f8r,f4",
                     help=f"Comma-separated dtypes ({','.join(VALID_DTYPES)})")
    ap.add_argument("--coco-dir", default=DEFAULT_COCO_DIR,
                     help="Path to COCO dataset root")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-images", type=int, default=0,
                     help="Limit number of images (0 = all)")
    ap.add_argument("--warmup", type=int, default=5,
                     help="Warmup iterations for speed benchmark")
    ap.add_argument("--iters", type=int, default=10,
                     help="Measured iterations for speed benchmark")
    ap.add_argument("--conf-thres", type=float, default=0.001,
                     help="Confidence threshold for NMS")
    ap.add_argument("--iou-thres", type=float, default=0.6,
                     help="IoU threshold for NMS")
    ap.add_argument("--json-output", type=Path, default=None,
                     help="Write results to JSON file")
    ap.add_argument("--memory", action="store_true",
                     help="Measure RSS memory during inference (Linux only)")
    ap.add_argument("--short", action="store_true",
                     help="Minimal output: summary table only, no verbose COCOeval tables")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model_names = [m.strip() for m in args.models.split(",")]
    dtypes = [d.strip() for d in args.dtypes.split(",")]
    for d in dtypes:
        if d not in VALID_DTYPES:
            print(f"error: unknown dtype '{d}' (valid: {','.join(VALID_DTYPES)})")
            return 1

    # Load COCO
    print(f"Loading COCO from {args.coco_dir} ...")
    coco, images, cat_map = _load_coco(args.coco_dir)
    if args.max_images > 0:
        images = images[: args.max_images]
    print(f"  {len(images)} images, {len(cat_map)} classes")

    # Build image_id lookup
    img_id_to_idx = {img["id"]: idx for idx, img in enumerate(images)}

    all_results: dict[str, dict[str, Any]] = {}

    for model_name in model_names:
        print(f"\n{'=' * 70}")
        print(f"  MODEL: {model_name}")
        print(f"{'=' * 70}")

        cache_dir = Path(f"/tmp/fastnn-yolo-coco/{model_name}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = cache_dir / "model.onnx"

        # Export ONNX if needed
        pt_name = MODEL_SPECS[model_name]["pt_name"]
        if not onnx_path.exists():
            print(f"  exporting {pt_name} -> {onnx_path}")
            _export_yolo_pt_to_onnx(pt_name, onnx_path, args.imgsz)

        # Load PyTorch model
        print(f"  loading {pt_name} ...")
        from ultralytics import YOLO
        pt_model = YOLO(pt_name)

        # --- PyTorch reference ---
        print(f"\n  Running PyTorch reference ...")
        pytorch_dets: list[tuple[int, np.ndarray]] = []
        pytorch_times: list[float] = []
        pt_mem = _MemoryTracker(enabled=args.memory, label="PyTorch")
        pt_mem.baseline()

        for img_info in images:
            img_path = str(Path(args.coco_dir) / "val2017" / img_info["file_name"])
            x_np, gain, pad = _preprocess(img_path, args.imgsz)
            x_batch = x_np[np.newaxis]  # [1, 3, 640, 640]

            t0 = time.perf_counter()
            y_pt = _run_pytorch(pt_model, x_batch)
            pytorch_times.append((time.perf_counter() - t0) * 1000.0)
            pt_mem.sample()

            dets_list = _postprocess(y_pt, args.conf_thres, args.iou_thres)
            dets = dets_list[0] if dets_list else np.empty((0, 6))
            dets = _scale_to_original(dets, (img_info["height"], img_info["width"]), gain, pad)
            pytorch_dets.append((img_info["id"], dets))

        pytorch_coco_results = _preds_to_coco_json(pytorch_dets, cat_map, img_id_to_idx)
        eval_img_ids = [img["id"] for img in images]
        pytorch_metrics = _evaluate_coco(coco, pytorch_coco_results, eval_img_ids,
                                     quiet=args.short)
        pytorch_speed = {
            "mean_ms": statistics.mean(pytorch_times),
            "std_ms": statistics.stdev(pytorch_times) if len(pytorch_times) > 1 else 0.0,
        }
        pytorch_mem_stats = pt_mem.summary()

        print(f"  PyTorch mAP@0.5={pytorch_metrics['mAP@0.5']:.4f}  "
              f"mAP@0.5:0.95={pytorch_metrics['mAP@0.5:0.95']:.4f}  "
              f"speed={pytorch_speed['mean_ms']:.1f}ms/img")
        if pytorch_mem_stats:
            print(f"  PyTorch RSS: baseline={pytorch_mem_stats['baseline_rss_mb']:.1f}MB  "
                  f"peak={pytorch_mem_stats['peak_rss_mb']:.1f}MB  "
                  f"delta={pytorch_mem_stats['delta_rss_mb']:.1f}MB  "
                  f"per_img={pytorch_mem_stats['per_image_mb']:.3f}MB")

        model_results: dict[str, Any] = {
            "pytorch": {
                "metrics": pytorch_metrics,
                "speed": pytorch_speed,
                "memory": pytorch_mem_stats,
                "num_predictions": len(pytorch_coco_results),
            },
            "fastnn": {},
        }

        # --- fastnn dtypes ---
        for dtype in dtypes:
            if not args.short:
                print(f"\n  --- dtype={dtype} ---")
            try:
                fastnn_dets: list[tuple[int, np.ndarray]] = []
                fastnn_times: list[float] = []

                if not args.short:
                    print(f"\n  Building fastnn executor for dtype={dtype} ...")
                executor, input_name, output_name = _make_fastnn_executor(onnx_path, dtype)
                fn_mem = _MemoryTracker(enabled=args.memory, label=f"fastnn/{dtype}")
                fn_mem.baseline()

                for img_info in images:
                    img_path = str(Path(args.coco_dir) / "val2017" / img_info["file_name"])
                    x_np, gain, pad = _preprocess(img_path, args.imgsz)
                    x_batch = x_np[np.newaxis]

                    import fastnn as fnn
                    fx = fnn.tensor(x_batch, list(x_batch.shape))

                    t0 = time.perf_counter()
                    result = executor.forward({input_name: fx})
                    y_fn = result[output_name].numpy()
                    fastnn_times.append((time.perf_counter() - t0) * 1000.0)
                    fn_mem.sample()

                    dets_list = _postprocess(y_fn, args.conf_thres, args.iou_thres)
                    dets = dets_list[0] if dets_list else np.empty((0, 6))
                    dets = _scale_to_original(dets, (img_info["height"], img_info["width"]), gain, pad)
                    fastnn_dets.append((img_info["id"], dets))

                fastnn_coco_results = _preds_to_coco_json(fastnn_dets, cat_map, img_id_to_idx)
                fastnn_metrics = _evaluate_coco(coco, fastnn_coco_results, eval_img_ids,
                                    quiet=args.short)
                fastnn_speed = {
                    "mean_ms": statistics.mean(fastnn_times),
                    "std_ms": statistics.stdev(fastnn_times) if len(fastnn_times) > 1 else 0.0,
                }
                fastnn_mem_stats = fn_mem.summary()

                # Compute delta vs PyTorch
                delta_mAP5 = fastnn_metrics["mAP@0.5"] - pytorch_metrics["mAP@0.5"]
                delta_mAP595 = fastnn_metrics["mAP@0.5:0.95"] - pytorch_metrics["mAP@0.5:0.95"]
                speedup = pytorch_speed["mean_ms"] / max(fastnn_speed["mean_ms"], 0.001)

                print(f"  mAP@0.5={fastnn_metrics['mAP@0.5']:.4f}  "
                      f"mAP@0.5:0.95={fastnn_metrics['mAP@0.5:0.95']:.4f}  "
                      f"speed={fastnn_speed['mean_ms']:.1f}ms/img  "
                      f"speedup={speedup:.2f}x")
                print(f"  ΔmAP@0.5={delta_mAP5:+.4f}  ΔmAP@0.5:0.95={delta_mAP595:+.4f}")
                if fastnn_mem_stats:
                    print(f"  RSS: baseline={fastnn_mem_stats['baseline_rss_mb']:.1f}MB  "
                          f"peak={fastnn_mem_stats['peak_rss_mb']:.1f}MB  "
                          f"delta={fastnn_mem_stats['delta_rss_mb']:.1f}MB  "
                          f"per_img={fastnn_mem_stats['per_image_mb']:.3f}MB")

                model_results["fastnn"][dtype] = {
                    "metrics": fastnn_metrics,
                    "speed": fastnn_speed,
                    "memory": fastnn_mem_stats,
                    "delta_vs_pytorch": {
                        "mAP@0.5": delta_mAP5,
                        "mAP@0.5:0.95": delta_mAP595,
                    },
                    "speedup": speedup,
                    "num_predictions": len(fastnn_coco_results),
                }

            except Exception as exc:
                import traceback
                print(f"  ERROR: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                model_results["fastnn"][dtype] = {"error": f"{type(exc).__name__}: {exc}"}

        all_results[model_name] = model_results

    # --- Summary table ---
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")

    for model_name, mr in all_results.items():
        pt = mr["pytorch"]
        print(f"\n  {model_name}:")
        pt_line = (f"    PyTorch:  mAP@0.5={pt['metrics']['mAP@0.5']:.4f}  "
                   f"mAP@0.5:0.95={pt['metrics']['mAP@0.5:0.95']:.4f}  "
                   f"{pt['speed']['mean_ms']:.1f}ms/img")
        if pt.get("memory"):
            pt_line += f"  RSS Δ={pt['memory']['delta_rss_mb']:.1f}MB"
        print(pt_line)

        for dtype, fr in mr["fastnn"].items():
            if "error" in fr:
                print(f"    {dtype:6s}: ERROR {fr['error']}")
            else:
                m = fr["metrics"]
                s = fr["speed"]
                d = fr["delta_vs_pytorch"]
                fn_line = (f"    {dtype:6s}: mAP@0.5={m['mAP@0.5']:.4f}  "
                           f"mAP@0.5:0.95={m['mAP@0.5:0.95']:.4f}  "
                           f"{s['mean_ms']:.1f}ms/img ({fr['speedup']:.2f}x)  "
                           f"Δ={d['mAP@0.5']:+.4f}/{d['mAP@0.5:0.95']:+.4f}")
                if fr.get("memory"):
                    fn_line += f"  RSS Δ={fr['memory']['delta_rss_mb']:.1f}MB"
                print(fn_line)

    # --- JSON output ---
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(all_results, indent=2, sort_keys=True))
        print(f"\nResults written to {args.json_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
