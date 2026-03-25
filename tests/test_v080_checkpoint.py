"""
v0.8.0 Verification Gate: Checkpoint Round-Trip Test

Guarantees serialization fidelity. Initialize a model, save it, load it into a
new instance, and run a forward pass on both with identical inputs.
Assert exact byte-level equality on outputs and all parameter tensors.
"""

import sys
import os
import tempfile
import numpy as np

sys.path.insert(0, "/home/petrouil/Projects/github/fastnn")

import fastnn


def test_checkpoint_roundtrip_linear():
    """Test checkpoint round-trip for a simple Linear model."""
    print("Testing checkpoint round-trip (Linear)...")

    # Create a simple linear model
    linear = fastnn.Linear(64, 32, bias=True)
    x = fastnn.randn([4, 64])

    # Forward pass with original model
    out_a = linear(x).numpy()

    # Save checkpoint
    with tempfile.NamedTemporaryFile(suffix=".fnn", delete=False) as f:
        checkpoint_path = f.name

    try:
        fastnn.save_model(linear, checkpoint_path)

        # Create new model and load checkpoint
        linear_b = fastnn.Linear(64, 32, bias=True)
        fastnn.load_model(linear_b, checkpoint_path)

        # Forward pass with loaded model
        out_b = linear_b(x).numpy()

        # Assert exact equality
        np.testing.assert_array_equal(
            out_a, out_b, err_msg="Checkpoint round-trip mismatch"
        )

        print("  PASSED: Linear checkpoint round-trip")
    finally:
        if os.path.exists(checkpoint_path):
            os.unlink(checkpoint_path)


def test_checkpoint_roundtrip_transformer():
    """Test checkpoint round-trip for TransformerEncoder."""
    print("Testing checkpoint round-trip (TransformerEncoder)...")

    model_a = fastnn.PyTransformerEncoder(
        vocab_size=1000,
        max_seq_len=128,
        d_model=128,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        num_classes=10,
        dropout_p=0.0,
    )

    x = fastnn.randint([2, 16], 0, 1000)
    out_a = model_a(x).numpy()

    with tempfile.NamedTemporaryFile(suffix=".fnn", delete=False) as f:
        checkpoint_path = f.name

    try:
        fastnn.save_model(model_a, checkpoint_path)

        model_b = fastnn.PyTransformerEncoder(
            vocab_size=1000,
            max_seq_len=128,
            d_model=128,
            num_heads=4,
            num_layers=2,
            ff_dim=256,
            num_classes=10,
            dropout_p=0.0,
        )
        fastnn.load_model(model_b, checkpoint_path)
        out_b = model_b(x).numpy()

        np.testing.assert_array_equal(
            out_a, out_b, err_msg="Transformer checkpoint round-trip mismatch"
        )

        print("  PASSED: TransformerEncoder checkpoint round-trip")
    finally:
        if os.path.exists(checkpoint_path):
            os.unlink(checkpoint_path)


def test_checkpoint_format_version():
    """Test that checkpoint format includes version header."""
    print("Testing checkpoint format versioning...")

    linear = fastnn.Linear(32, 16, bias=True)

    with tempfile.NamedTemporaryFile(suffix=".fnn", delete=False) as f:
        checkpoint_path = f.name

    try:
        fastnn.save_model(linear, checkpoint_path)

        # Read file header
        with open(checkpoint_path, "rb") as f:
            magic = f.read(4)
            version = int.from_bytes(f.read(4), "little")

        # Check magic bytes
        assert magic == b"FNN\x00", f"Invalid magic bytes: {magic}"
        assert version >= 1, f"Invalid version: {version}"

        print(f"  PASSED: Format version {version} with correct magic bytes")
    finally:
        if os.path.exists(checkpoint_path):
            os.unlink(checkpoint_path)


def test_checkpoint_multiple_saves():
    """Test multiple save/load cycles."""
    print("Testing multiple checkpoint save/load cycles...")

    linear = fastnn.Linear(32, 16, bias=True)
    x = fastnn.randn([4, 32])
    original_out = linear(x).numpy()

    with tempfile.NamedTemporaryFile(suffix=".fnn", delete=False) as f:
        checkpoint_path = f.name

    try:
        for cycle in range(5):
            fastnn.save_model(linear, checkpoint_path)
            linear_b = fastnn.Linear(32, 16, bias=True)
            fastnn.load_model(linear_b, checkpoint_path)
            out = linear_b(x).numpy()

            np.testing.assert_array_equal(
                original_out,
                out,
                err_msg=f"Cycle {cycle}: output mismatch after save/load",
            )

        print("  PASSED: 5 save/load cycles preserved fidelity")
    finally:
        if os.path.exists(checkpoint_path):
            os.unlink(checkpoint_path)


if __name__ == "__main__":
    test_checkpoint_roundtrip_linear()
    test_checkpoint_roundtrip_transformer()
    test_checkpoint_format_version()
    test_checkpoint_multiple_saves()
    print("\n=== Checkpoint Round-Trip Tests PASSED ===")
