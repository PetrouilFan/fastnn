#!/usr/bin/env python3
"""COCO mAP evaluation for YOLO26 models via fastnn.

Measures detection accuracy (mAP@0.5:0.95, mAP@0.5) across all data types
using the COCO val2017 dataset.

Usage::

    # F32 baseline on yolo26n
    python scripts/yolo_fastnn_coco_eval.py --model yolo26n --dtype f32 \\
        --coco-dir D:/data/coco

    # All dtypes for both models
    python scripts/yolo_fastnn_coco_eval.py \\
        --models yolo26n,yolo26l --dtype f32,u4,u8,f8,f8r,f4 \\
        --coco-dir D:/data/coco

    # Also run PyTorch reference
    python scripts/yolo_fastnn_coco_eval.py --model yolo26n --coco-dir D:/data/coco \\
        --run-pytorch

    # Save results
    python scripts/yolo_fastnn_coco_eval.py --model yolo26n \\
        --coco-dir D:/data/coco --output /tmp/coco_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Known model metadata (mirrors yolo_compare_fastnn_pytorch.py)
# ---------------------------------------------------------------------------

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "yolo26n": {
        "hf_repo": "onnx-community/yolo26n-ONNX",
        "hf_file": "onnx/model.onnx",
        "pt_name": "yolo26n.pt",
    },
    "yolo26l": {
        "hf_repo": "onnx-community/yolo26l-ONNX",
        "hf_file": "onnx/model.onnx",
        "pt_name": "yolo26l.pt",
    },
}

VALID_DTYPES = ("f32", "u4", "u8", "i4", "i8", "f8", "f8r", "f4")
COCO_CATEGORIES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]
COCO_NUM_CLASSES = 80
# YOLO26 output: (1, N, 6) = [x1, y1, x2, y2, conf, cls]
YOLO26_MAX_DET = 300


# ---------------------------------------------------------------------------
# Letterbox preprocessing
# ---------------------------------------------------------------------------

def letterbox(
    im: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, float, float]:
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, dw, dh


# ---------------------------------------------------------------------------
# COCO dataset loader
# ---------------------------------------------------------------------------


def load_coco_annotations(ann_path: str) -> dict[str, Any]:
    with open(ann_path) as f:
        return json.load(f)


def _coco_image_ids(annotations: dict[str, Any]) -> list[int]:
    return [img["id"] for img in annotations["images"]]


def _image_path_by_id(annotations: dict[str, Any], img_id: int, coco_img_dir: str) -> str | None:
    for img in annotations["images"]:
        if img["id"] == img_id:
            return os.path.join(coco_img_dir, img["file_name"])
    return None


# ---------------------------------------------------------------------------
# Preprocess single image for inference
# ---------------------------------------------------------------------------


def preprocess_image(image_path: str, imgsz: int) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    img0 = cv2.imread(image_path)
    if img0 is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

    img, r, dw, dh = letterbox(img0, new_shape=(imgsz, imgsz))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)[np.newaxis, ...]  # NCHW
    return img, img0, r, dw, dh


# ---------------------------------------------------------------------------
# Postprocess YOLO26 detections
# ---------------------------------------------------------------------------


def postprocess_detections(
    raw_output: np.ndarray,
    img0_shape: tuple[int, int],
    ratio_pad: tuple[float, tuple[float, float]],
    conf_threshold: float = 0.001,
) -> np.ndarray:
    raw_output = raw_output.reshape(-1, 6)

    mask = raw_output[:, 4] >= conf_threshold
    dets = raw_output[mask]

    if dets.shape[0] == 0:
        return np.empty((0, 6), dtype=np.float32)

    gain, (pad_x, pad_y) = ratio_pad
    dets[:, 0] = (dets[:, 0] - pad_x) / gain
    dets[:, 1] = (dets[:, 1] - pad_y) / gain
    dets[:, 2] = (dets[:, 2] - pad_x) / gain
    dets[:, 3] = (dets[:, 3] - pad_y) / gain

    dets[:, [0, 2]] = dets[:, [0, 2]].clip(0, img0_shape[1])
    dets[:, [1, 3]] = dets[:, [1, 3]].clip(0, img0_shape[0])
    return dets


def detections_to_coco_json(
    detections: np.ndarray,
    image_id: int,
) -> list[dict[str, Any]]:
    results = []
    for i in range(detections.shape[0]):
        x1, y1, x2, y2, conf, cls_id = detections[i].tolist()
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue
        results.append({
            "image_id": image_id,
            "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
            "score": round(float(conf), 6),
            "category_id": int(cls_id) + 1,
        })
    return results


# ---------------------------------------------------------------------------
# ONNX / fastnn helpers
# ---------------------------------------------------------------------------


def _dtype_to_quantize(dtype: str) -> int | str | None:
    if dtype == "f32":
        return None
    if dtype == "u4" or dtype == "u8":
        return dtype  # unsigned → pass as string for WeightDtype::U4/U8
    if dtype == "i4":
        return 4  # signed → pass as int for WeightDtype::I4
    if dtype == "i8":
        return 8  # signed → pass as int for WeightDtype::I8
    return dtype


def _download_hf_onnx(model_name: str, onnx_path: Path) -> None:
    import urllib.request

    spec = MODEL_SPECS[model_name]
    url = f"https://huggingface.co/{spec['hf_repo']}/resolve/main/{spec['hf_file']}"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    urllib.request.urlretrieve(url, str(onnx_path))
    print(f"  saved -> {onnx_path}")


def _make_fastnn_executor(onnx_path: Path, dtype: str):
    import onnx
    from onnx import numpy_helper

    import fastnn as fnn
    from fastnn.io.onnx import _extract_attrs

    model = onnx.shape_inference.infer_shapes(onnx.load(str(onnx_path)))
    initializer_names = {init.name for init in model.graph.initializer}
    input_names = [i.name for i in model.graph.input if i.name not in initializer_names]
    output_names = [o.name for o in model.graph.output]
    known_shapes: dict[str, list[int]] = {}
    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        dims = []
        tensor_type = vi.type.tensor_type
        if tensor_type.HasField("shape"):
            for d in tensor_type.shape.dim:
                dv = d.dim_value if d.HasField("dim_value") else 0
                dims.append(int(dv) if dv > 0 else 1)
            known_shapes[vi.name] = dims
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
                    const_values[out_name] = np.take(const_values[node.input[0]], const_values[node.input[1]].astype(np.int64), axis=axis)
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
        if node.output and node.output[0] in known_shapes:
            item["output_shape"] = ",".join(str(d) for d in known_shapes[node.output[0]])
        if node.op_type == "Resize" and len(node.input) >= 3:
            scales_name = node.input[2]
            if scales_name in params:
                scales = np.asarray(params[scales_name].numpy(), dtype=np.float32).reshape(-1)
                if scales.size >= 4:
                    item["scale_h"] = str(int(scales[2]))
                    item["scale_w"] = str(int(scales[3]))
        for k, v in attrs.items():
            if k == "value" or k in item:
                continue
            item[k] = v.decode("utf-8") if isinstance(v, bytes) else str(v)
        nodes.append(item)

    quantize = _dtype_to_quantize(dtype)
    executor = fnn.AotExecutor(nodes, params, input_names, output_names, input_shapes=input_shapes, quantize=quantize)
    return executor, input_names[0], output_names[0]


# ---------------------------------------------------------------------------
# PyTorch runner (optional reference)
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


def _run_pytorch(model: Any, x: np.ndarray) -> np.ndarray:
    import torch

    torch.set_num_threads(1)
    model.model.eval()
    xt = torch.from_numpy(x)
    with torch.no_grad():
        y = model.model(xt)
    if isinstance(y, (tuple, list)):
        y = y[0]
    if hasattr(y, "detach"):
        y = y.detach().cpu().numpy()
    return y


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def evaluate_model(
    onnx_path: Path,
    dtype: str,
    coco_ann: dict[str, Any],
    coco_img_dir: str,
    imgsz: int,
    conf_threshold: float,
    max_images: int | None,
    run_pytorch: bool = False,
) -> dict[str, Any]:
    print(f"\n  [dtype={dtype}] building fastnn executor ...")
    executor, input_name, output_name = _make_fastnn_executor(onnx_path, dtype=dtype)
    image_ids = _coco_image_ids(coco_ann)
    if max_images is not None:
        image_ids = image_ids[:max_images]

    all_results: list[dict[str, Any]] = []
    total_images = len(image_ids)

    if run_pytorch:
        pt_model = _export_yolo_pt_to_onnx(
            MODEL_SPECS[onnx_path.parent.name] if onnx_path.parent.name in MODEL_SPECS else onnx_path.stem,
            onnx_path,
            imgsz,
        )

    t0 = time.perf_counter()
    for idx, img_id in enumerate(image_ids):
        img_path = _image_path_by_id(coco_ann, img_id, coco_img_dir)
        if img_path is None:
            continue

        try:
            x, img0, r, dw, dh = preprocess_image(img_path, imgsz)
        except FileNotFoundError:
            continue

        import fastnn as fnn

        fx = fnn.tensor(x, list(x.shape))
        fy = executor.forward({input_name: fx})[output_name].numpy()

        img0_h, img0_w = img0.shape[:2]
        ratio_pad = (r, (dw, dh))

        dets = postprocess_detections(fy, (img0_h, img0_w), ratio_pad, conf_threshold=conf_threshold)
        results_json = detections_to_coco_json(dets, img_id)
        all_results.extend(results_json)

        if (idx + 1) % 500 == 0 or idx == 0 or idx == total_images - 1:
            elapsed = time.perf_counter() - t0
            print(f"    [{idx+1}/{total_images}] {len(all_results)} detections  ({elapsed:.1f}s)")

    total_time = time.perf_counter() - t0

    result = {
        "dtype": dtype,
        "num_images": total_images,
        "num_detections": len(all_results),
        "total_time_s": round(total_time, 3),
        "avg_time_per_image_ms": round(total_time / max(total_images, 1) * 1000, 3),
    }

    print(f"    [dtype={dtype}] evaluating mAP with pycocotools ...")
    result["coco_metrics"] = compute_coco_metrics(coco_ann, all_results)

    if run_pytorch:
        print(f"    [dtype={dtype}] running PyTorch reference ...")
        pt_results: list[dict[str, Any]] = []
        for idx, img_id in enumerate(image_ids):
            img_path = _image_path_by_id(coco_ann, img_id, coco_img_dir)
            if img_path is None:
                continue
            try:
                x, img0, r, dw, dh = preprocess_image(img_path, imgsz)
            except FileNotFoundError:
                continue
            y_pt = _run_pytorch(pt_model, x)
            img0_h, img0_w = img0.shape[:2]
            ratio_pad = (r, (dw, dh))
            dets = postprocess_detections(y_pt, (img0_h, img0_w), ratio_pad, conf_threshold=conf_threshold)
            pt_results.extend(detections_to_coco_json(dets, img_id))
            if (idx + 1) % 500 == 0:
                print(f"      pt [{idx+1}/{total_images}]")
        result["pytorch_metrics"] = compute_coco_metrics(coco_ann, pt_results)

    return result


def compute_coco_metrics(
    coco_ann: dict[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, float]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO()
    coco_gt.dataset = coco_ann
    coco_gt.createIndex()

    if not results:
        return {
            "mAP@0.50:0.95": 0.0,
            "mAP@0.50": 0.0,
            "mAP@0.75": 0.0,
            "mAP_small": 0.0,
            "mAP_medium": 0.0,
            "mAP_large": 0.0,
            "AR@1": 0.0,
            "AR@10": 0.0,
            "AR@100": 0.0,
        }

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    return {
        "mAP@0.50:0.95": round(float(stats[0]), 4),
        "mAP@0.50": round(float(stats[1]), 4),
        "mAP@0.75": round(float(stats[2]), 4),
        "mAP_small": round(float(stats[3]), 4),
        "mAP_medium": round(float(stats[4]), 4),
        "mAP_large": round(float(stats[5]), 4),
        "AR@1": round(float(stats[6]), 4),
        "AR@10": round(float(stats[7]), 4),
        "AR@100": round(float(stats[8]), 4),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="YOLO26 COCO mAP evaluation via fastnn"
    )
    ap.add_argument("--models", default="yolo26n",
                    help="Comma-separated model names (default: yolo26n)")
    ap.add_argument("--dtype", default="f32",
                    help=f"Comma-separated dtypes ({','.join(VALID_DTYPES)}, default: f32)")
    ap.add_argument("--coco-dir", required=True,
                    help="COCO dataset root containing val2017/ and annotations/instances_val2017.json")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="Inference image size (default: 640)")
    ap.add_argument("--conf", type=float, default=0.001,
                    help="Confidence threshold (default: 0.001 for COCO eval)")
    ap.add_argument("--max-images", type=int, default=None,
                    help="Limit number of images for quick smoke test")
    ap.add_argument("--output", type=Path, default=None,
                    help="Save results JSON to this path")
    ap.add_argument("--run-pytorch", action="store_true",
                    help="Also run PyTorch reference for comparison")
    args = ap.parse_args()

    coco_img_dir = os.path.join(args.coco_dir, "val2017")
    coco_ann_path = os.path.join(args.coco_dir, "annotations", "instances_val2017.json")

    for p in (coco_img_dir, coco_ann_path):
        if not os.path.exists(p):
            print(f"error: required path not found: {p}")
            print("Download COCO val2017 from https://cocodataset.org/#download")
            return 1

    print(f"COCO images: {coco_img_dir}")
    print(f"COCO annotations: {coco_ann_path}")
    coco_ann = load_coco_annotations(coco_ann_path)
    total_available = len(coco_ann["images"])
    print(f"Total images in annotation: {total_available}")

    model_names = [m.strip() for m in args.models.split(",")]
    dtypes = [d.strip() for d in args.dtype.split(",")]
    for d in dtypes:
        if d not in VALID_DTYPES:
            print(f"error: unknown dtype '{d}'")
            return 1

    results_rows: list[dict[str, Any]] = []

    for model_name in model_names:
        print(f"\n{'=' * 70}")
        print(f"  MODEL: {model_name}")
        print(f"{'=' * 70}")

        cache_dir = Path(f"/tmp/fastnn-yolo-verify/{model_name}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = cache_dir / "model.onnx"

        if not onnx_path.exists():
            print("  downloading ONNX from HuggingFace ...")
            _download_hf_onnx(model_name, onnx_path)
        else:
            print(f"  using cached ONNX: {onnx_path}")

        for dtype in dtypes:
            print(f"\n  --- dtype={dtype} ---")
            try:
                res = evaluate_model(
                    onnx_path,
                    dtype,
                    coco_ann,
                    coco_img_dir,
                    imgsz=args.imgsz,
                    conf_threshold=args.conf,
                    max_images=args.max_images,
                    run_pytorch=args.run_pytorch and dtype == dtypes[0],
                )
                results_rows.append({
                    "model": model_name,
                    **res,
                })
                metrics = res.get("coco_metrics", {})
                print(f"    mAP@0.50:0.95={metrics.get('mAP@0.50:0.95', '?'):>8}")
                print(f"    mAP@0.50     ={metrics.get('mAP@0.50', '?'):>8}")
                print(f"    avg time     ={res['avg_time_per_image_ms']:.1f} ms")
                if "pytorch_metrics" in res:
                    pt = res["pytorch_metrics"]
                    print(f"    PT mAP@0.50:0.95={pt.get('mAP@0.50:0.95', '?'):>8}")
                    print(f"    PT mAP@0.50     ={pt.get('mAP@0.50', '?'):>8}")
            except Exception as exc:
                import traceback
                print(f"    ERROR: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                results_rows.append({
                    "model": model_name,
                    "dtype": dtype,
                    "error": f"{type(exc).__name__}: {exc}",
                })

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    for row in results_rows:
        model = row["model"]
        dtype = row["dtype"]
        if "error" in row:
            print(f"  {model} {dtype:6s}: ERROR {row['error']}")
        else:
            m = row.get("coco_metrics", {})
            avg_ms = row["avg_time_per_image_ms"]
            print(f"  {model} {dtype:6s}: mAP50-95={m.get('mAP@0.50:0.95', 0):.4f}  "
                  f"mAP50={m.get('mAP@0.50', 0):.4f}  {avg_ms:.1f}ms/img")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results_rows, indent=2, sort_keys=True))
        print(f"\nResults written to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
