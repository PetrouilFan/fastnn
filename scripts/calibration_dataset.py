#!/usr/bin/env python3
"""
Calibration Dataset Loader for YOLO Quantization

Loads COCO validation images, applies YOLO preprocessing (letterbox, normalize),
and yields batches for calibration data collection.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple, Optional, List
import random

try:
    import torch
    import torchvision.transforms as T
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def letterbox_image(image: np.ndarray, target_size: Tuple[int, int], 
                    stride: int = 32, auto: bool = True) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """
    Resize and pad image while meeting stride-multiple constraints.
    
    Returns:
        (letterboxed_image, (scale_w, scale_h), (pad_w, pad_h))
    """
    shape = image.shape[:2]  # [h, w]
    target_h, target_w = target_size
    
    # Scale ratio (new / old)
    r = min(target_h / shape[0], target_w / shape[1])
    if not auto:  # only scale down, don't scale up
        r = min(r, 1.0)
    
    # Compute padding
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = target_w - new_unpad[0], target_h - new_unpad[1]
    
    if auto:  # ensure stride multiple
        dw = np.mod(dw, stride)
        dh = np.mod(dh, stride)
    
    dw /= 2
    dh /= 2
    
    if shape[::-1] != new_unpad:  # resize
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, 
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    return image, (r, r), (dw, dh)


class CalibrationDataset:
    """Generator for calibration images from COCO dataset."""
    
    def __init__(
        self,
        data_root: str = "/data/coco",
        split: str = "val2017",
        target_size: Tuple[int, int] = (320, 320),
        max_samples: int = 500,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.target_size = target_size
        self.max_samples = max_samples
        self.shuffle = shuffle
        self.seed = seed
        
        # Find images
        img_dir = self.data_root / split
        if not img_dir.exists():
            # Try common alternative locations
            for alt in ["images", "val2017", "train2017"]:
                alt_dir = self.data_root / alt
                if alt_dir.exists():
                    img_dir = alt_dir
                    break
        
        self.image_paths = list(img_dir.glob("*.jpg"))
        if not self.image_paths:
            self.image_paths = list(img_dir.glob("*.png"))
        
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {img_dir}")
        
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(self.image_paths)
        
        self.image_paths = self.image_paths[:self.max_samples]
        print(f"CalibrationDataset: {len(self.image_paths)} images from {img_dir}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield preprocessed images as FP32 NCHW tensors."""
        for img_path in self.image_paths:
            # Load image (BGR -> RGB)
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Letterbox to target size
            image, _, _ = letterbox_image(image, self.target_size)
            
            # Convert to FP32 NCHW, normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            image = image.transpose(2, 0, 1)  # HWC -> CHW
            image = np.expand_dims(image, 0)  # Add batch dim: [1, C, H, W]
            
            yield image
    
    def get_batch(self, batch_size: int) -> Iterator[np.ndarray]:
        """Yield batches of images."""
        batch = []
        for img in self:
            batch.append(img)
            if len(batch) >= batch_size:
                yield np.concatenate(batch, axis=0)
                batch = []
        if batch:
            yield np.concatenate(batch, axis=0)


def get_calibration_dataloader(
    data_root: str = "/data/coco",
    split: str = "val2017",
    target_size: Tuple[int, int] = (320, 320),
    max_samples: int = 500,
    batch_size: int = 1,
) -> Tuple[CalibrationDataset, Iterator[np.ndarray]]:
    """
    Get calibration dataset and batched iterator.
    
    Returns:
        (dataset, batched_iterator)
    """
    dataset = CalibrationDataset(
        data_root=data_root,
        split=split,
        target_size=target_size,
        max_samples=max_samples,
    )
    return dataset, dataset.get_batch(batch_size)


def save_calibration_images(
    output_dir: str,
    data_root: str = "/data/coco",
    split: str = "val2017",
    target_size: Tuple[int, int] = (320, 320),
    max_samples: int = 500,
) -> None:
    """Save preprocessed calibration images as .npy files for fast loading."""
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = CalibrationDataset(data_root, split, target_size, max_samples)
    for i, img in enumerate(dataset):
        np.save(os.path.join(output_dir, f"calib_{i:04d}.npy"), img)
        if i % 100 == 0:
            print(f"Saved {i+1}/{len(dataset)} calibration images")
    
    print(f"Saved {len(dataset)} calibration images to {output_dir}")


def load_calibration_images(input_dir: str) -> List[np.ndarray]:
    """Load preprocessed calibration images from .npy files."""
    paths = sorted(Path(input_dir).glob("calib_*.npy"))
    images = [np.load(p) for p in paths]
    print(f"Loaded {len(images)} calibration images from {input_dir}")
    return images


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate calibration dataset")
    parser.add_argument("--data-root", default="/data/coco", help="COCO dataset root")
    parser.add_argument("--split", default="val2017", help="Dataset split")
    parser.add_argument("--target-size", type=int, nargs=2, default=[320, 320], help="Target H W")
    parser.add_argument("--max-samples", type=int, default=500, help="Max images")
    parser.add_argument("--output-dir", help="Save as .npy files")
    parser.add_argument("--test", action="store_true", help="Test loading")
    
    args = parser.parse_args()
    
    if args.output_dir:
        save_calibration_images(
            args.output_dir,
            args.data_root,
            args.split,
            tuple(args.target_size),
            args.max_samples,
        )
    elif args.test:
        dataset = CalibrationDataset(
            args.data_root, args.split, tuple(args.target_size), args.max_samples
        )
        for i, img in enumerate(dataset):
            print(f"Image {i}: shape={img.shape}, range=[{img.min():.4f}, {img.max():.4f}]")
            if i >= 5:
                break
    else:
        # Just test iterator
        dataset = CalibrationDataset(
            args.data_root, args.split, tuple(args.target_size), args.max_samples
        )
        print(f"Dataset length: {len(dataset)}")
        for i, batch in enumerate(dataset.get_batch(4)):
            print(f"Batch {i}: shape={batch.shape}")
            if i >= 2:
                break