# Optimizers

FastNN provides several optimization algorithms for training neural networks.

## SGD (Stochastic Gradient Descent)

```python
import fastnn as fnn

optimizer = fnn.SGD(
    parameters=model.parameters(),
    lr=0.01,              # Learning rate (required)
    momentum=0.0,         # Momentum factor
    weight_decay=0.0     # L2 regularization
)

# Training loop
for batch in loader:
    optimizer.zero_grad()      # Clear gradients
    loss = compute_loss()      # Forward pass
    loss.backward()            # Backward pass
    optimizer.step()           # Update parameters
```

**Parameters:**
- `parameters`: Iterable of parameters to optimize
- `lr`: Learning rate
- `momentum`: Momentum factor (default: 0)
- `weight_decay`: Weight decay/L2 regularization (default: 0)

## Adam

Adaptive Moment Estimation optimizer.

```python
optimizer = fnn.Adam(
    parameters=model.parameters(),
    lr=0.001,             # Learning rate
    betas=(0.9, 0.999),   # Beta1, Beta2 for momentum
    eps=1e-8,             # Small constant for numerical stability
    weight_decay=0.0     # L2 regularization
)
```

**Parameters:**
- `parameters`: Iterable of parameters to optimize
- `lr`: Learning rate (default: 0.001)
- `betas`: Coefficients for computing running averages (default: (0.9, 0.999))
- `eps`: Term for numerical stability (default: 1e-8)
- `weight_decay`: Weight decay (default: 0)

## AdamW

Adam optimizer with weight decay correction.

```python
optimizer = fnn.AdamW(
    parameters=model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01  # AdamW applies decoupled weight decay
)
```

**Note:** AdamW applies weight decay differently than Adam with `weight_decay`. It decouples weight decay from the gradient update, leading to better generalization.

## Training Loop Pattern

```python
import fastnn as fnn

# Setup
model = fnn.models.MLP(input_dim=784, hidden_dims=[256, 128], output_dim=10)
optimizer = fnn.Adam(model.parameters(), lr=1e-3)
loss_fn = fnn.mse_loss  # or fnn.cross_entropy_loss

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        x, y = batch
        
        # Forward pass
        pred = model(x)
        loss = loss_fn(pred, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Update
        optimizer.step()
    
    # Print progress
    print(f"Epoch {epoch}: loss = {loss.item():.4}")
```

## Gradient Clipping

To prevent exploding gradients:

```python
# Clip gradients by global norm
max_norm = 1.0
total_norm = 0.0
for p in model.parameters():
    if p.grad is not None:
        param_norm = (p.grad ** 2).sum()
        total_norm += param_norm.item()
total_norm = total_norm ** 0.5

if total_norm > max_norm:
    clip_factor = max_norm / total_norm
    for p in model.parameters():
        if p.grad is not None:
            p.grad *= clip_factor
```

## Learning Rate Scheduling

Adjust learning rate during training using callbacks:

```python
from fastnn.callbacks import LearningRateScheduler

# Step decay
lr_scheduler = LearningRateScheduler(
    schedule="step",
    lr=0.01,
    step_size=10,
    gamma=0.1  # Multiply lr by 0.1 every step_size epochs
)

# Or cosine annealing
lr_scheduler = LearningRateScheduler(
    schedule="cosine",
    lr=0.01,
    T_max=100,
    eta_min=0.0001
)

# In training loop
for epoch in range(100):
    # ... training code ...
    lr_scheduler.on_epoch_end(epoch, logs)
    current_lr = logs.get("lr", optimizer.lr)
```

## Choosing an Optimizer

| Optimizer | Use Case |
|-----------|----------|
| SGD | Simple baseline, often with momentum |
| Adam | Default choice for most tasks |
| AdamW | When weight decay is important (recommended for transformers) |
