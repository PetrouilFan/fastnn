"""
Autonomous benchmark: fastnn vs PyTorch CPU inference.
Downloads model automatically, no user input required.
Includes both inference speed and ImageNet accuracy evaluation.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import json
import fastnn as fnn
from fastnn.export import export_pytorch_model, load_fnn_model

# Models to benchmark
MODELS = [
    ("resnet18", torchvision.models.resnet18),
    ("mobilenet_v2", torchvision.models.mobilenet_v2),
]

# Benchmark config
BATCH_SIZES = [1, 4, 16]
WARMUP_ITERS = 20
BENCH_ITERS = 100
INPUT_SHAPE = (3, 224, 224)

# ImageNet config
IMAGENET_SUBSET_SIZE = 500  # Use subset for faster evaluation (set to None for full dataset)


def benchmark_pytorch(model, input_tensor, warmup, iters):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
        times = []
        for _ in range(iters):
            start = time.perf_counter()
            _ = model(input_tensor)
            times.append(time.perf_counter() - start)
    return times


def benchmark_fastnn(model, input_tensor, warmup, iters):
    # input_tensor is a fastnn tensor
    for _ in range(warmup):
        _ = model(input_tensor)
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        _ = model(input_tensor)
        times.append(time.perf_counter() - start)
    return times


def evaluate_accuracy_pytorch(model, dataloader, device="cpu"):
    """Evaluate PyTorch model accuracy on ImageNet."""
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Top-1 accuracy
            _, pred_top1 = outputs.topk(1, dim=1)
            correct_top1 += pred_top1.eq(labels.view(-1, 1)).sum().item()

            # Top-5 accuracy
            _, pred_top5 = outputs.topk(5, dim=1)
            correct_top5 += pred_top5.eq(labels.view(-1, 1).expand_as(pred_top5)).sum().item()

            total += labels.size(0)

    top1_acc = 100.0 * correct_top1 / total
    top5_acc = 100.0 * correct_top5 / total
    return top1_acc, top5_acc, total


def evaluate_accuracy_fastnn(model, dataloader):
    """Evaluate fastnn model accuracy on ImageNet."""
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for images, labels in dataloader:
        # Convert PyTorch tensor to fastnn tensor
        batch_size = images.shape[0]
        images_np = images.numpy().astype(np.float32)
        fnn_input = fnn.tensor(images_np.flatten().tolist(), list(images_np.shape))

        # Forward pass
        outputs = model(fnn_input)

        # Convert fastnn output to numpy for evaluation
        outputs_np = np.array(outputs.data).reshape(batch_size, -1)

        # Top-1 accuracy
        pred_top1 = np.argmax(outputs_np, axis=1)
        correct_top1 += np.sum(pred_top1 == labels.numpy())

        # Top-5 accuracy
        top5_preds = np.argsort(outputs_np, axis=1)[:, -5:]
        for i in range(batch_size):
            if labels.numpy()[i] in top5_preds[i]:
                correct_top5 += 1

        total += batch_size

    top1_acc = 100.0 * correct_top1 / total
    top5_acc = 100.0 * correct_top5 / total
    return top1_acc, top5_acc, total


def get_imagenet_dataloader(batch_size=32, subset_size=None):
    """Load ImageNet validation dataset with standard preprocessing."""
    # Standard ImageNet preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        # Try to load ImageNet validation dataset
        dataset = torchvision.datasets.ImageNet(
            root="/tmp/imagenet",
            split="val",
            transform=transform,
        )

        # Use subset if specified
        if subset_size is not None and subset_size < len(dataset):
            indices = np.random.choice(len(dataset), subset_size, replace=False)
            dataset = torch.utils.data.Subset(dataset, indices)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        return dataloader

    except Exception as e:
        print(f"Warning: Could not load ImageNet dataset: {e}")
        print("Using synthetic data for accuracy evaluation...")
        return None


def create_synthetic_dataloader(batch_size=32, num_samples=100):
    """Create synthetic data for testing when ImageNet is not available."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create synthetic images and labels
    images = torch.randn(num_samples, 3, 224, 224)
    labels = torch.randint(0, 1000, (num_samples,))

    dataset = torch.utils.data.TensorDataset(images, labels)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return dataloader


def run_speed_benchmark(model_name, pt_model, fnn_model):
    """Run inference speed benchmark."""
    print(f"\n  Speed Benchmark:")
    results = {}

    for batch_size in BATCH_SIZES:
        # Create identical inputs
        np_input = np.random.randn(batch_size, *INPUT_SHAPE).astype(np.float32)

        # PyTorch input
        pt_input = torch.tensor(np_input)

        # fastnn input
        fnn_input = fnn.tensor(np_input.flatten().tolist(), list(np_input.shape))

        # Run benchmarks
        pt_times = benchmark_pytorch(pt_model, pt_input, WARMUP_ITERS, BENCH_ITERS)
        fnn_times = benchmark_fastnn(fnn_model, fnn_input, WARMUP_ITERS, BENCH_ITERS)

        pt_mean = np.mean(pt_times) * 1000  # ms
        fnn_mean = np.mean(fnn_times) * 1000
        speedup = pt_mean / fnn_mean

        key = f"batch{batch_size}"
        results[key] = {
            "pytorch_ms": round(pt_mean, 3),
            "fastnn_ms": round(fnn_mean, 3),
            "speedup": round(speedup, 3),
            "faster": "fastnn" if speedup > 1 else "pytorch",
        }

        print(
            f"    batch={batch_size}: PyTorch={pt_mean:.2f}ms  fastnn={fnn_mean:.2f}ms  speedup={speedup:.2f}x"
        )

    return results


def run_accuracy_benchmark(model_name, pt_model, fnn_model):
    """Run ImageNet accuracy benchmark."""
    print(f"\n  Accuracy Benchmark (ImageNet):")

    # Load ImageNet validation data
    dataloader = get_imagenet_dataloader(batch_size=32, subset_size=IMAGENET_SUBSET_SIZE)

    if dataloader is None:
        print("    Using synthetic data (not real ImageNet)")
        dataloader = create_synthetic_dataloader(batch_size=32, num_samples=100)

    # Evaluate PyTorch model
    print("    Evaluating PyTorch model...")
    pt_top1, pt_top5, pt_total = evaluate_accuracy_pytorch(pt_model, dataloader)
    print(f"      PyTorch: Top-1={pt_top1:.2f}%, Top-5={pt_top5:.2f}% ({pt_total} images)")

    # Evaluate fastnn model
    print("    Evaluating fastnn model...")
    fnn_top1, fnn_top5, fnn_total = evaluate_accuracy_fastnn(fnn_model, dataloader)
    print(f"      fastnn:  Top-1={fnn_top1:.2f}%, Top-5={fnn_top5:.2f}% ({fnn_total} images)")

    # Calculate accuracy difference
    top1_diff = fnn_top1 - pt_top1
    top5_diff = fnn_top5 - pt_top5

    results = {
        "pytorch_top1": round(pt_top1, 2),
        "pytorch_top5": round(pt_top5, 2),
        "fastnn_top1": round(fnn_top1, 2),
        "fastnn_top5": round(fnn_top5, 2),
        "top1_diff": round(top1_diff, 2),
        "top5_diff": round(top5_diff, 2),
        "num_images": pt_total,
    }

    print(f"      Difference: Top-1={top1_diff:+.2f}%, Top-5={top5_diff:+.2f}%")

    return results


def run_all_benchmarks():
    results = {}
    for model_name, model_fn in MODELS:
        print(f"\n{'='*70}")
        print(f"Benchmarking {model_name}")
        print(f"{'='*70}")

        # Download pretrained PyTorch model
        print("  Loading pretrained PyTorch model...")
        pt_model = model_fn(pretrained=True)
        pt_model.eval()

        # Export to .fnn format
        print("  Exporting to .fnn format...")
        export_path = f"/tmp/{model_name}.fnn"
        export_pytorch_model(pt_model, export_path, input_shape=INPUT_SHAPE)

        # Load as fastnn model
        print("  Loading fastnn model...")
        fnn_model = load_fnn_model(export_path)

        # Run speed benchmark
        speed_results = run_speed_benchmark(model_name, pt_model, fnn_model)

        # Run accuracy benchmark
        accuracy_results = run_accuracy_benchmark(model_name, pt_model, fnn_model)

        # Combine results
        results[model_name] = {
            "speed": speed_results,
            "accuracy": accuracy_results,
        }

    # Save results
    with open("tests/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print("Benchmark Complete!")
    print(f"{'='*70}")
    print("\nResults saved to tests/benchmark_results.json")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for model_name in results:
        print(f"\n{model_name}:")
        speed = results[model_name]["speed"]
        acc = results[model_name]["accuracy"]

        # Speed summary
        for batch_key in speed:
            batch_data = speed[batch_key]
            print(f"  Speed (batch={batch_key.replace('batch', '')}): "
                  f"PyTorch={batch_data['pytorch_ms']:.2f}ms, "
                  f"fastnn={batch_data['fastnn_ms']:.2f}ms, "
                  f"speedup={batch_data['speedup']:.2f}x")

        # Accuracy summary
        print(f"  Accuracy: PyTorch Top-1={acc['pytorch_top1']:.2f}%, "
              f"fastnn Top-1={acc['fastnn_top1']:.2f}% "
              f"(diff={acc['top1_diff']:+.2f}%)")
        print(f"  Accuracy: PyTorch Top-5={acc['pytorch_top5']:.2f}%, "
              f"fastnn Top-5={acc['fastnn_top5']:.2f}% "
              f"(diff={acc['top5_diff']:+.2f}%)")

    return results


if __name__ == "__main__":
    run_all_benchmarks()
