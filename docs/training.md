# Training

FastNN provides utilities for building training pipelines: datasets, data loaders, and callbacks.

## Dataset

Base class for custom datasets:

```python
import fastnn as fnn

class MyDataset(fnn.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Usage
dataset = MyDataset(x_data, y_labels)
sample = dataset[0]  # Returns (data, label) tuple
```

## TensorDataset

Convenience class for tensor data:

```python
import fastnn as fnn

# Create tensors
X = fnn.randn(1000, 784)  # 1000 samples, 784 features
y = fnn.randint(0, 10, (1000,))  # 1000 labels (0-9)

# Wrap in dataset
dataset = fnn.TensorDataset(X, y)

# Access
sample = dataset[0]  # Returns (X[0], y[0])
print(len(dataset))  # 1000
```

## DataLoader

Batches and shuffles data for training:

```python
import fastnn as fnn

dataset = fnn.TensorDataset(X, y)

loader = fnn.DataLoader(
    dataset,
    batch_size=32,      # Samples per batch
    shuffle=True,       # Random shuffle each epoch
    drop_last=False,    # Drop incomplete final batch
    num_workers=0       # Parallel data loading (0 = disabled)
)

# Iterate
for batch_x, batch_y in loader:
    # batch_x: (batch_size, 784)
    # batch_y: (batch_size,)
    predictions = model(batch_x)
    # ... training code ...
```

**Parameters:**
- `dataset`: Dataset to load from
- `batch_size`: Number of samples per batch
- `shuffle`: Whether to shuffle each epoch
- `drop_last`: Drop last incomplete batch
- `num_workers`: Number of worker processes

## Training Loop

Complete training loop example:

```python
import fastnn as fnn
from fastnn.callbacks import EarlyStopping, ModelCheckpoint

# Setup
model = fnn.models.MLP(input_dim=784, hidden_dims=[256, 128], output_dim=10)
optimizer = fnn.Adam(model.parameters(), lr=1e-3)
loss_fn = fnn.cross_entropy_loss

# Callbacks
early_stopping = EarlyStopping(
    monitor="loss",
    patience=5,
    min_delta=0.001
)

# Training
model.train()
for epoch in range(100):
    epoch_loss = 0
    for batch_x, batch_y in loader:
        # Forward
        pred = model(batch_x)
        loss = loss_fn(pred, batch_y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(loader)
    print(f"Epoch {epoch}: loss = {avg_loss:.4}")
    
    # Check early stopping
    early_stopping.on_epoch_end(epoch, {"loss": avg_loss})
```

## Callbacks

### EarlyStopping

Stops training when monitored metric stops improving:

```python
from fastnn.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor="val_loss",           # Metric to monitor
    patience=5,                   # Epochs to wait before stopping
    min_delta=0.0,                # Minimum change to qualify as improvement
    restore_best_weights=True    # Restore best model weights
)
```

### ModelCheckpoint

Saves model when monitored metric improves:

```python
from fastnn.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    dir_path="./checkpoints/",
    monitor="val_loss",
    mode="min",                   # "min" or "max"
    save_best_only=True,
    verbose=True
)
```

### LearningRateScheduler

Adjusts learning rate during training:

```python
from fastnn.callbacks import LearningRateScheduler

# Step decay
lr_scheduler = LearningRateScheduler(
    schedule="step",
    lr=0.01,
    step_size=10,
    gamma=0.1
)

# Or cosine annealing
lr_scheduler = LearningRateScheduler(
    schedule="cosine",
    lr=0.01,
    T_max=100,
    eta_min=0.0001
)
```

### CSVLogger

Logs metrics to CSV file:

```python
from fastnn.callbacks import CSVLogger

csv_logger = CSVLogger(filepath="./logs/training.csv")
```

## Loss Functions

```python
# Mean Squared Error
loss = fnn.mse_loss(predictions, targets)

# Cross Entropy (for classification)
loss = fnn.cross_entropy_loss(logits, targets)
```

## Inference

Use `no_grad` context to disable gradient computation during inference:

```python
model.eval()  # Set to evaluation mode

with fnn.no_grad():
    predictions = model(test_data)
    
# Or using context manager
with fnn.no_grad():
    for sample in test_loader:
        output = model(sample)
```
