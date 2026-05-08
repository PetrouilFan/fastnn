"""Test DataLoader with multiple workers."""
import time
import fastnn as fnn

# Create test data
x = fnn.randn([32, 3, 32, 32])
y = fnn.randint(0, 10, [32])  # Fixed: randint(low, high, shape)
ds = fnn.data.TensorDataset(x, y)

print('Testing DataLoader with 0 workers...')
start = time.perf_counter()
loader = fnn.data.DataLoader(ds, batch_size=4, num_workers=0)
count = 0
for batch in loader:
    count += 1
print(f'  Got {count} batches in {time.perf_counter()-start:.2f}s')

print('\nTesting DataLoader with 2 workers...')
start = time.perf_counter()
loader = fnn.data.DataLoader(ds, batch_size=4, num_workers=2)
count = 0
for batch in loader:
    count += 1
print(f'  Got {count} batches in {time.perf_counter()-start:.2f}s')

print('\n✓ DataLoader test completed')
