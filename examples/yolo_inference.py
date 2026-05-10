#!/usr/bin/env python3
"""Example: YOLO object detection with fastnn ONNX pipeline.

This example demonstrates:
1. Loading a YOLO model from ONNX format
2. Running inference on an image
3. Displaying detection results

Usage:
    python examples/yolo_inference.py --model yolov8n.onnx --image image.jpg
    python examples/yolo_inference.py --model yolov11n.onnx --image image.jpg --conf 0.5
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO inference with fastnn")
    parser.add_argument("--model", "-m", required=True, help="Path to ONNX model")
    parser.add_argument("--image", "-i", required=True, help="Path to input image")
    parser.add_argument("--conf", "-c", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--max-det", type=int, default=100, help="Max detections per image")
    parser.add_argument("--classes", "-cls", nargs="+", type=int, default=None,
                        help="Filter by class IDs (e.g., 0 for person)")
    parser.add_argument("--output", "-o", default=None, help="Output image path with boxes drawn")
    parser.add_argument("--benchmark", "-b", action="store_true", help="Run benchmark mode")
    return parser.parse_args()


def benchmark_model(model_path: str, num_iters: int = 50):
    """Benchmark model inference speed."""
    import fastnn as fnn

    print(f"\nBenchmarking {model_path}...")
    print(f"  Running {num_iters} iterations")

    # Load model once
    start = time.time()
    model = fnn.YOLO(str(model_path))
    load_time = time.time() - start
    print(f"  Model load time: {load_time:.3f}s")

    # Create random input at model's expected size
    dummy_input = np.random.randn(640, 640, 3).astype(np.float32) * 255

    # Warmup
    for _ in range(3):
        _ = model(dummy_input)

    # Benchmark
    start = time.time()
    for _ in range(num_iters):
        _ = model(dummy_input)
    elapsed = time.time() - start
    avg_ms = (elapsed / num_iters) * 1000

    print(f"  Average inference time: {avg_ms:.1f}ms per image")
    print(f"  Throughput: {1000/avg_ms:.1f} FPS")
    print()


def draw_detections(image_path: str, detections: np.ndarray, output_path: str):
    """Draw detection boxes on an image and save."""
    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        colors = [
            "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF",
            "#00FFFF", "#FF8000", "#8000FF", "#0080FF", "#FF0080",
        ]

        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det[:6]
            color = colors[int(cls_id) % len(colors)]

            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            label = f"cls{int(cls_id)} {conf:.2f}"
            bbox = draw.textbbox((x1, y1), label)
            draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], fill=color)
            draw.text((x1, y1), label, fill="white")

        img.save(output_path)
        print(f"  Output saved to: {output_path}")
    except ImportError:
        print("  PIL not installed. Skipping visualization.")
        print(f"  Detections: {len(detections)} objects found")


def main():
    args = parse_args()

    if args.benchmark:
        benchmark_model(args.model)
        return

    import fastnn as fnn

    print(f"Loading model: {args.model}")
    start = time.time()

    model = fnn.YOLO(
        str(args.model),
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        max_detections=args.max_det,
        classes=args.classes,
    )

    load_time = time.time() - start
    print(f"  Model loaded in {load_time:.3f}s")

    print(f"Running inference on: {args.image}")
    start = time.time()
    detections = model(args.image)
    infer_time = time.time() - start

    print(f"  Inference time: {infer_time*1000:.1f}ms")

    for i, img_dets in enumerate(detections):
        print(f"\n  Image {i}: {len(img_dets)} detections")
        if len(img_dets) > 0:
            print(f"  {'x1':>6} {'y1':>6} {'x2':>6} {'y2':>6} {'conf':>5} {'cls':>3}")
            print(f"  {'-'*35}")
            for det in img_dets[:10]:
                x1, y1, x2, y2, conf, cls_id = det[:6]
                print(f"  {x1:>6.1f} {y1:>6.1f} {x2:>6.1f} {y2:>6.1f} {conf:>5.2f} {int(cls_id):>3}")
            if len(img_dets) > 10:
                print(f"  ... and {len(img_dets) - 10} more")

    if args.output and len(detections) > 0 and len(detections[0]) > 0:
        draw_detections(args.image, detections[0], args.output)


if __name__ == "__main__":
    main()
