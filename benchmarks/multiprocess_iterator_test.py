"""Test MultiProcessIterator with multiple workers."""
import time
import fastnn as fnn
from fastnn.data import DataLoader, TensorDataset

def slow_fetch(idx):
    """Simulate slow data loading."""
    time.sleep(0.01)  # 10ms delay
    return idx

class SlowDataset:
    def __len__(self):
        return 32
    
    def __getitem__(self, idx):
        slow_fetch(idx)
        return fnn.randn([3, 32, 32]), idx

if __name__ == "__main__":
    dataset = SlowDataset()
    
    # Test with 0 workers (sequential)
    print("Testing DataLoader with 0 workers...")
    loader = DataLoader(dataset, batch_size=4, num_workers=0)
    start = time.perf_counter()
    batches = list(loader)
    elapsed_0 = time.perf_counter() - start
    print(f"  0 workers: {elapsed_0:.2f}s for {len(batches)} batches")
    
    # Test with 2 workers
    print("Testing DataLoader with 2 workers...")
    loader = DataLoader(dataset, batch_size=4, num_workers=2)
    start = time.perf_counter()
    batches = list(loader)
    elapsed_2 = time.perf_counter() - start
    print(f"  2 workers: {elapsed_2:.2f}s for {len(batches)} batches")
    
    print(f"\nSpeedup with 2 workers: {elapsed_0/elapsed_2:.1f}x")
    print("✓ MultiProcessIterator test completed")
