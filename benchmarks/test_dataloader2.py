"""Test DataLoader with multiple workers."""
import time
import fastnn as fnn
import numpy as np

# Create test data using numpy and convert to fastnn tensors
x_np = np.random.randn(32, 3, 32, 32).astype(np.float32)
y_np = np.random.randint(0, 10, 32)

x = fnn.tensor(x_np, [32, 3, 32, 32])
y = fnn.tensor(y_np, [32])

ds = fnn.data.TensorDataset(x, y)

print('Testing DataLoader with 0 workers...')
start = time.time()
loader = fnn.data.DataLoader(ds, batch_size=4, num_workers=0)
count = 0
for batch in loader:
    count += 1
print(f'  Got {count} batches in {time.time()-start:.2f}s')

print('\nTesting DataLoader with 2 workers...')
start = time.time()
loader = fnn.data.DataLoader(ds, batch_size=4, num_workers=2)
count = 0
for batch in loader:
    count += 1
print(f'  Got {count} batches in {time.time()-start:.2f}s')

print('\n✓ DataLoader test completed')
