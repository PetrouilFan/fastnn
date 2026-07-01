# Training Basics

fastnn supports two training modes: **eager mode** (imperative, for research and prototyping) and **compiled training** (AOT-compiled forward+backward+optimizer pipeline for production workloads).

## Eager Mode Training

Eager mode mirrors the PyTorch training loop pattern:

```python
import fastnn as fnn

model = fnn.Sequential(
    fnn.Linear(784, 256),
    fnn.ReLU(),
    fnn.Linear(256, 10),
)
optimizer = fnn.Adam(model.parameters(), lr=1e-3)

model.train()
for epoch in range(10):
    for batch_x, batch_y in dataloader:
        pred = model(batch_x)
        loss = fnn.cross_entropy_loss(pred, batch_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### Training Loop Steps

1. **Forward pass**: `model(batch_x)` computes predictions
2. **Loss**: `loss_fn(pred, target)` computes scalar loss
3. **Backward pass**: `loss.backward()` accumulates gradients
4. **Optimizer step**: `optimizer.step()` updates parameters
5. **Zero gradients**: `optimizer.zero_grad()` clears gradients for next iteration

### Model Modes

```python
model.train()   # Enables dropout, batch norm training behavior
model.eval()    # Disables dropout, uses running stats for batch norm
```

## Compiled Training (AOT)

For production training, fastnn can compile the entire forward+backward+optimizer pipeline into a single `ExecutablePlan`:

```python
from fastnn import compile_train_model

model = compile_train_model(
    graph_bytes=graph_bytes,
    loss_node_id=loss_id,
    param_ids=[param_id_1, param_id_2],
    param_data=[weight_bytes_1, weight_bytes_2],
    batch_input_ids=[input_id],
    optimizer="adamw",
    lr=0.001,
    weight_decay=0.01,
)

# Single dispatch per training step
loss = model.train_step([batch_input_bytes])
```

### Supported Optimizers

| String | Optimizer | Key Parameters |
|--------|-----------|----------------|
| `"sgd"` | SGD | lr, weight_decay |
| `"adamw"` | AdamW | lr, beta1, beta2, eps, weight_decay |
| `"muon"` | Muon | lr, weight_decay |
| `"lion"` | Lion | lr, beta1, beta2 |
| `"rmsprop"` | RMSprop | lr, beta, eps |

Compiled training eliminates Python overhead and runs the entire step as a single compiled kernel dispatch. See [Compiled Training API](../../reference/python-api.md) for full parameter details.

## Data Loading

```python
from fastnn.data import DataLoader

loader = DataLoader(dataset, batch_size=32, shuffle=True)
for batch_x, batch_y in loader:
    # training step
```

## Loss Functions

```python
loss = fnn.mse_loss(pred, target)        # Mean squared error
loss = fnn.cross_entropy_loss(pred, target)  # Cross entropy
loss = fnn.binary_cross_entropy(pred, target) # Binary cross entropy
loss = fnn.l1_loss(pred, target)         # L1 loss
```

## See Also

- [Callbacks & Checkpointing](callbacks.md) — saving, logging, and metrics
- [Distributed Training](distributed.md) — multi-device training
- [Optimizers Reference](../../reference/optimizers.md) — optimizer documentation
- [Getting Started](../getting-started.md) — first program
