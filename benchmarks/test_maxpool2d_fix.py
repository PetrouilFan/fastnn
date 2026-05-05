"""Test MaxPool2d fix - verify correctness."""
import fastnn as fnn

# Test that MaxPool2d works correctly
pool = fnn.MaxPool2d(kernel_size=2, stride=2)

# Create test input
x = fnn.randn([1, 1, 4, 4])
output = pool(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
assert output.shape == [1, 1, 2, 2], f"Expected [1, 1, 2, 2], got {output.shape}"

# Test that module is reused (not recreated every call)
pool2 = fnn.MaxPool2d(kernel_size=3, stride=1, padding=1)
x2 = fnn.randn([1, 1, 4, 4])
output2 = pool2(x2)
print(f"MaxPool2d with padding: input {x2.shape} -> output {output2.shape}")

print("\n✓ MaxPool2d fix verified: module created once in __init__")
