# Training

## Compiled Training (v2.2)

FastNN v2.2 supports **compiled training** — the entire forward+backward+optimizer
pipeline is compiled ahead of time into an `ExecutablePlan` that executes
with a single dispatch call, reusing a persistent memory arena across steps.

### Supported Optimizers

| Optimizer | Opcode | State | Compiled Support |
|-----------|--------|-------|-----------------|
| SGD       | SgdUpdate | None | ✅ |
| Adam      | AdamUpdate | m, v | ✅ |
| AdamW     | AdamWUpdate | m, v | ✅ |
| Muon      | MuonUpdate | None | ✅ (v2.2) |
| Lion      | LionUpdate | m | ✅ (v2.2) |
| RMSprop   | RmspropUpdate | v | ✅ (v2.2) |

### Python API

```python
import fastnn as fnn

# Build your model graph (using the GraphBuilder or ONNX)
# Serialize to bytes
graph_bytes = ...

# Compile for training
model = fnn.compile_train_model(
    graph_bytes=graph_bytes,
    loss_node_id=...,
    param_ids=[...],
    param_data=[...],
    batch_input_ids=[...],
    optimizer="adamw",       # "sgd", "adamw", "muon", "lion", "rmsprop"
    lr=0.001,
    weight_decay=0.01,
)

# Training loop
for epoch in range(10):
    for batch in dataloader:
        loss = model.train_step([batch_bytes])
        print(f"loss = {loss:.4f}")
```

### Rust API

```rust
use fastnn::backend::cpu::CpuBackend;
use fastnn::backend::executor::GraphExecutor;
use fastnn::compiler::passes::training::{OptimizerConfig, TrainConfig};
use fastnn::ir::builder::GraphBuilder;

// Build graph
let g = GraphBuilder::new();
let x = g.input(&[1, 4], IrDType::F32);
let w = g.parameter(&[4, 2], IrDType::F32);
let mm = g.matmul(&x, &w);
let loss = g.reduce_mean(&mm, 0, false);
let graph = g.to_graph();
let param_data = vec![w_bytes];

// Compile
let executor = GraphExecutor::new(CpuBackend);
let mut model = executor.compile_train(
    &graph,
    loss.node_id(),
    &[w.node_id()],
    &param_data,
    &[x.node_id()],
    None,
    &TrainConfig {
        optimizer: OptimizerConfig::AdamW {
            lr: 0.001, beta1: 0.9, beta2: 0.999,
            eps: 1e-8, weight_decay: 0.01,
        },
        quantize: None,
    },
).unwrap();

// Train
for step in 0..100 {
    let loss = model.train_step(&[batch_bytes]).unwrap();
    println!("Step {}: loss = {}", step, loss);
}
```

### Shape Tightening

When using dynamic batch dimensions, pass concrete shapes via
`batch_shape_env` to tighten memory allocation:

```rust
let mut shape_env = ShapeEnv::new();
shape_env.bind("batch", 2);
shape_env.bind("seq", 128);

let mut model = executor.compile_train(
    &graph, loss_id, &params, &param_data, &inputs,
    Some(&shape_env),  // tighten with concrete shapes
    &config,
).unwrap();
```

This shrinks the memory arena from worst-case (85+ GB) to actual sizes (~40 KB).

---

## Eager Mode Training

FastNN also provides utilities for building training pipelines: datasets, data loaders, optimizers, LR schedulers, and callbacks.

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
X = fnn.randn([1000, 784])  # 1000 samples, 784 features
y = fnn.randint(low=0, high=10, shape=[1000])  # 1000 labels (0-9)

# Wrap in dataset
dataset = fnn.TensorDataset(X, y)

# Access
sample = dataset[0]  # Returns (X[0], y[0])
print(len(dataset))  # 1000
```

## DataLoader

Batches, shuffles, and prefetches data for training:

```python
import fastnn as fnn

dataset = fnn.TensorDataset(X, y)

loader = fnn.DataLoader(
    dataset,
    batch_size=32,       # Samples per batch
    shuffle=True,        # Random shuffle each epoch
    drop_last=False,     # Drop incomplete final batch
    prefetch_size=2,     # Number of batches to prefetch ahead
)

# Iterate
for batch_x, batch_y in loader:
    # batch_x: [batch_size, 784]
    # batch_y: [batch_size]
    predictions = model(batch_x)
    # ... training code ...
```

**Parameters:**
- `dataset`: Dataset to load from
- `batch_size`: Number of samples per batch
- `shuffle`: Whether to shuffle each epoch
- `drop_last`: Drop last incomplete batch
- `prefetch_size`: Number of batches to prefetch (default: 2)
- `collate_fn`: Custom collation function (default: default_collate)

### Samplers

```python
from fastnn.data import SequentialSampler, RandomSampler, SubsetRandomSampler, BatchSampler

# Sequential
sampler = SequentialSampler(dataset)

# Random with custom generator
sampler = RandomSampler(dataset, generator=random.Random(42))

# Subset
sampler = SubsetRandomSampler([0, 10, 20, 30])

# Batch sampler
batch_sampler = BatchSampler(sampler, batch_size=32, drop_last=True)
```

### Resetting Sampler (New Epoch)

```python
for epoch in range(100):
    loader.reset_sampler()  # Re-shuffle for new epoch
    for batch_x, batch_y in loader:
        ...
```

## Optimizers

### Adam

```python
optimizer = fnn.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0,
)
```

### AdamW

```python
optimizer = fnn.AdamW(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
)
```

### SGD

```python
optimizer = fnn.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0,
    nesterov=False,
)
```

### RMSprop

```python
optimizer = fnn.RMSprop(
    model.parameters(),
    lr=0.01,
    alpha=0.99,
    eps=1e-8,
    weight_decay=0.0,
    momentum=0.0,
    centered=False,
)
```

### Muon

```python
optimizer = fnn.Muon(
    model.parameters(),
    lr=0.025,
    momentum=0.95,
    weight_decay=0.0,
    nesterov=True,
)
```

### Lion

```python
optimizer = fnn.Lion(
    model.parameters(),
    lr=0.0001,
    betas=(0.95, 0.98),
    weight_decay=0.0,
)
```

## Learning Rate Schedulers

### StepLR

Decays LR by gamma every step_size epochs:

```python
scheduler = fnn.StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(100):
    for batch_x, batch_y in loader:
        ...
    scheduler.step()  # Updates optimizer LR
```

### CosineAnnealingLR

```python
scheduler = fnn.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

for epoch in range(100):
    ...
    scheduler.step()
```

### ExponentialLR

```python
scheduler = fnn.ExponentialLR(optimizer, gamma=0.95)
```

### ReduceLROnPlateau

Reduces LR when a metric stops improving:

```python
scheduler = fnn.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=10,
    min_lr=1e-6,
)

for epoch in range(100):
    ...
    val_loss = evaluate()
    scheduler.step(val_loss)
```

## Gradient Clipping

```python
# Clip by norm
fnn.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2.0)

# Clip by value
fnn.clip_grad_value_(model.parameters(), clip_value=0.1)
```

## Loss Functions

```python
# Mean squared error
loss = fnn.mse_loss(pred, target)

# Cross-entropy (with logits)
loss = fnn.cross_entropy_loss(logits, target)

# Binary cross-entropy with logits
loss = fnn.bce_with_logits(logits, target)

# Huber loss (smooth L1)
loss = fnn.huber_loss(pred, target, delta=1.0)
```

## Callbacks

### EarlyStopping

```python
early_stop = fnn.EarlyStopping(
    monitor='val_loss',
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
)
```

### ModelCheckpoint

```python
checkpoint = fnn.ModelCheckpoint(
    dir_path='checkpoints/',
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=True,
)
```

### CSVLogger

```python
logger = fnn.CSVLogger(filepath='training_log.csv')
```

## Model I/O

### Unified API (fastnn.io)

```python
import fastnn as fnn

# Save/Load models (custom binary format .fnn)
fnn.io.save(model, "model.fnn")
loaded_model = fnn.io.load("model.fnn")

# Convert from other formats
fnn.io.convert_from_pytorch(torch_model, "model.fnn")
info = fnn.io.convert_from_onnx("model.onnx", "model.fnn")
```


## Complete Training Example

```python
import fastnn as fnn

# Create model
model = fnn.models.MLP(input_dim=784, hidden_dims=[256, 128], output_dim=10)

# Create optimizer
optimizer = fnn.Adam(model.parameters(), lr=0.001)

# Create LR scheduler
scheduler = fnn.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# Create data loader
ds = fnn.TensorDataset(x_train, y_train)
loader = fnn.DataLoader(ds, batch_size=64, shuffle=True, prefetch_size=2)

# Training loop
for epoch in range(50):
    loader.reset_sampler()  # Re-shuffle for new epoch
    for batch_x, batch_y in loader:
        # Forward
        logits = model(batch_x)
        loss = fnn.cross_entropy_loss(logits, batch_y)

        # Backward
        loss.backward()

        # Gradient clipping
        fnn.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update
        optimizer.step()
        optimizer.zero_grad()

    # Update LR
    scheduler.step()

    # Save checkpoint
    if epoch % 10 == 0:
        fnn.io.save(model, f'checkpoint_epoch_{epoch}.fnn')
```

## Distributed Data Parallel

FastNN provides `DataParallel` for multi-GPU training using data parallelism.

```python
from fastnn.parallel import DataParallel

# Create model replicas for each GPU
model_gpu0 = fnn.models.MLP(input_dim=784, hidden_dims=[256], output_dim=10)
model_gpu1 = fnn.models.MLP(input_dim=784, hidden_dims=[256], output_dim=10)

# Initialize DataParallel
dp_model = DataParallel(
    [model_gpu0, model_gpu1],
    device_ids=[0, 1],
    weights=[0.6, 0.4]  # Optional: data distribution weights
)

# Training loop
for x_batch, y_batch in loader:
    loss = dp_model.forward_backward(x_batch, y_batch, fnn.cross_entropy_loss)
    dp_model.sync_gradients()
    
    # Step optimizers for each replica
    for opt in optimizers:
        opt.step()
        opt.zero_grad()
```

Parameters:
- `models`: List of model replicas (one per GPU)
- `device_ids`: List of GPU device IDs (e.g., [0, 1])
- `weights`: Optional list of data weights per GPU (e.g., [0.6, 0.4] for uneven GPUs)

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
