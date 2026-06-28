#!/usr/bin/env python3
"""
YOLO Calibration Script - Runs calibration on YOLO model using fastnn Python API
and saves scales.json for quantized compilation.
"""

import sys
import os
import json
from pathlib import Path

# Headless: no GUI backend
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

# Add fastnn to path
sys.path.insert(0, "/home/petrouil/Projects/github/fastnn")

import fastnn as fnn
from fastnn.io.onnx import import_onnx
from scripts.calibration_dataset import CalibrationDataset

_COCO_ROOT = Path("/home/petrouil/data/coco")


def export_yolo_pt_to_onnx(pt_path, onnx_path, imgsz=320):
    """Export Ultralytics YOLO .pt to ONNX."""
    from ultralytics import YOLO
    model = YOLO(pt_path)
    model.export(format="onnx", imgsz=imgsz, opset=12, simplify=True, half=False)
    # Ultralytics saves to runs/detect/export/...
    exported = list(Path("runs/detect/export").rglob("*.onnx"))
    if not exported:
        # Try alternative location
        exported = list(Path(".").rglob(f"*{Path(pt_path).stem}*.onnx"))
    if exported:
        import shutil
        shutil.move(str(exported[0]), str(onnx_path))
    return onnx_path


def load_yolo_onnx_for_fastnn(onnx_path):
    """Load ONNX and convert to fastnn format using Python API."""
    import onnx
    from onnx import numpy_helper
    from fastnn.io.onnx import _extract_attrs
    
    model = onnx.shape_inference.infer_shapes(onnx.load(str(onnx_path)))
    initializer_names = {init.name for init in model.graph.initializer}
    input_names = [
        i.name for i in model.graph.input if i.name not in initializer_names
    ]
    output_names = [o.name for o in model.graph.output]
    
    input_shapes = {}
    for i in model.graph.input:
        if i.name in input_names:
            tensor_type = i.type.tensor_type
            if tensor_type.HasField("shape"):
                input_shapes[i.name] = [
                    int(d.dim_value) if d.HasField("dim_value") and d.dim_value > 0 else -1
                    for d in tensor_type.shape.dim
                ]
    
    initializers_by_name = {init.name: init for init in model.graph.initializer}
    
    params = {}
    for init in model.graph.initializer:
        arr_raw = numpy_helper.to_array(init)
        arr = arr_raw.astype(np.float32, copy=False)
        params[init.name] = fnn.tensor(arr, list(arr.shape))
    
    nodes = []
    for idx, node in enumerate(model.graph.node):
        name = node.name or f"{node.op_type}_{idx}"
        attrs = _extract_attrs(node)
        
        if node.op_type == "Gemm" and len(node.input) >= 2:
            transB = int(attrs.get("transB", 0))
            if transB == 1 and node.input[1] in params and node.input[1] in initializers_by_name:
                w_init = initializers_by_name[node.input[1]]
                w = numpy_helper.to_array(w_init)
                w_t = np.ascontiguousarray(w.T)
                params[node.input[1]] = fnn.tensor(
                    w_t.astype(np.float32, copy=False), list(w_t.shape)
                )
                attrs["transB"] = 0
        
        # Handle Resize scales for Ultralytics YOLO
        if node.op_type == "Resize" and len(node.input) >= 3:
            scales_name = node.input[2]
            if scales_name in params:
                scales = np.asarray(params[scales_name].numpy(), dtype=np.float32).reshape(-1)
                if scales.size >= 4:
                    attrs["scale_h"] = str(int(scales[2]))
                    attrs["scale_w"] = str(int(scales[3]))
        
        item = {
            "name": name,
            "op_type": node.op_type,
            "inputs": ",".join(node.input),
            "outputs": ",".join(node.output),
        }
        for k, v in attrs.items():
            if k == "value" or k in item:
                continue
            # Format attribute values
            if isinstance(v, (list, tuple)):
                item[k] = ",".join(str(x) for x in v)
            elif isinstance(v, float):
                item[k] = repr(v)
            elif isinstance(v, np.ndarray):
                item[k] = ",".join(str(x) for x in v.reshape(-1))
            else:
                item[k] = str(v)
        nodes.append(item)
    
    return nodes, params, input_names, output_names, input_shapes


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Calibrate YOLO model for fastnn quantization")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics YOLO .pt model")
    parser.add_argument("--onnx", default="/tmp/fastnn-yolo-verify/yolov8n.onnx", help="ONNX output path")
    parser.add_argument("--imgsz", type=int, default=320, help="Input image size")
    parser.add_argument("--coco", default="/home/petrouil/data/coco/val2017", help="COCO val2017 images")
    parser.add_argument("--annotations", default="/home/petrouil/data/coco/annotations/instances_val2017.json", help="COCO annotations")
    parser.add_argument("--num-samples", type=int, default=500, help="Number of calibration images")
    parser.add_argument("--bit-width", type=int, choices=[4, 8], default=8, help="Quantization bit width")
    parser.add_argument("--output", default="scales.json", help="Output scales.json path")
    args = parser.parse_args()

    # Check COCO availability early
    if not _COCO_ROOT.is_dir():
        print(f"COCO data not found at {_COCO_ROOT}")
        print("SKIP: calibration requires COCO val2017 images — place the dataset at /home/petrouil/data/coco")
        return 0

    onnx_path = Path(args.onnx)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting YOLO: {args.model} -> {onnx_path}")
    export_yolo_pt_to_onnx(args.model, onnx_path, args.imgsz)
    
    print("Loading ONNX for fastnn...")
    nodes, params, input_names, output_names, input_shapes = load_yolo_onnx_for_fastnn(onnx_path)
    
    print(f"Building AotExecutor with {len(nodes)} nodes...")
    executor = fnn.AotExecutor(nodes, params, input_names, output_names, input_shapes=input_shapes)
    
    print(f"Loading calibration dataset ({args.num_samples} images)...")
    calib_dataset = CalibrationDataset(
        data_root=args.coco,
        split="val2017" if "val2017" in args.coco else "train2017",
        target_size=(args.imgsz, args.imgsz),
        max_samples=args.num_samples,
        shuffle=True,
        seed=42,
    )
    
    calibration_inputs = []
    for i, img in enumerate(calib_dataset):
        calibration_inputs.append({input_names[0]: fnn.tensor(img, list(img.shape))})
        if i % 100 == 0:
            print(f"  Loaded {i+1}/{len(calib_dataset)} calibration images")
    
    print(f"Running calibration with {len(calibration_inputs)} images...")
    cal_data = executor.calibrate(calibration_inputs, bit_width=args.bit_width)
    print(f"Calibration complete! Found {len(cal_data)} calibration points.")
    
    # Save scales.json
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Format for CLI consumption
    scales_dict = {}
    for name, stats in cal_data.items():
        scales_dict[name] = {
            "scale": float(stats["scale"]),
            "zero_point": float(stats["zero_point"]),
            "bit_width": int(stats["bit_width"]),
            "min": float(stats["min"]),
            "max": float(stats["max"]),
            "mean": float(stats["mean"]),
            "std": float(stats["std"]),
        }
    
    with open(output_path, "w") as f:
        json.dump(scales_dict, f, indent=2)
    
    print(f"Scales saved to {output_path}")
    
    # Print summary
    print("\nCalibration Summary:")
    for name, stats in sorted(scales_dict.items(), key=lambda x: x[1]["scale"])[:20]:
        print(f"  {name}: scale={stats['scale']:.6f}, zp={stats['zero_point']:.1f}, range=[{stats['min']:.3f}, {stats['max']:.3f}]")


if __name__ == "__main__":
    main()