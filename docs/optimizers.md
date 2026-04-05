# Optimizers

FastNN provides several optimization algorithms for training neural networks.

## SGD (Stochastic Gradient Descent)

```python
import fastnn as fnn

optimizer = fnn.SGD(
    parameters=model.parameters(),
    lr=0.01,              # Learning rate (required)
    momentum=0.0,         # Momentum factor
    weight_decay=0.0,     # L2 regularization
    nesterov=False        # Use Nesterov momentum
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
- `nesterov`: Use Nesterov momentum (default: False)

## Adam

Adaptive Moment Estimation optimizer.

```python
optimizer = fnn.Adam(
    parameters=model.parameters(),
    lr=0.001,             # Learning rate
    betas=(0.9, 0.999),   # Beta1, Beta2 for momentum
    eps=1e-8,             # Small constant for numerical stability
    weight_decay=0.0      # L2 regularization
)
```

**Parameters:**
- `parameters`: Iterable of parameters to optimize
- `lr`: Learning rate (default: 0.001)
- `betas`: Coefficients for computing running averages (default: (0.9, 0.999))
- `eps`: Term for numerical stability (default: 1e-8)
- `weight_decay`: Weight decay (default: 0)

## AdamW

Adam optimizer with decoupled weight decay.

```python
optimizer = fnn.AdamW(
    parameters=model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01  # AdamW applies decoupled weight decay
)
```

**Note:** AdamW applies weight decay differently than Adam. It decouples weight decay from the gradient update, leading to better generalization.

## RMSprop

Root Mean Square Propagation optimizer.

```python
optimizer = fnn.RMSprop(
    parameters=model.parameters(),
    lr=0.01,
    alpha=0.99,           # Smoothing constant for squared gradient
    eps=1e-8,
    weight_decay=0.0,
    momentum=0.0,         # Momentum factor
    centered=False        # If True, use centered RMSprop
)
```

**Parameters:**
- `parameters`: Iterable of parameters to optimize
- `lr`: Learning rate (default: 0.01)
- `alpha`: Smoothing constant (default: 0.99)
- `eps`: Term for numerical stability (default: 1e-8)
- `weight_decay`: Weight decay (default: 0)
- `momentum`: Momentum factor (default: 0)
- `centered`: If True, compute centered RMSprop (default: False)

## Muon

Muon optimizer with orthogonalized momentum.

```python
optimizer = fnn.Muon(
    parameters=model.parameters(),
    lr=0.025,
    momentum=0.95,
    weight_decay=0.0,
    nesterov=True
)
```

**Parameters:**
- `parameters`: Iterable of parameters to optimize
- `lr`: Learning rate (default: 0.025)
- `momentum`: Momentum factor (default: 0.95)
- `weight_decay`: Weight decay (default: 0)
- `nesterov`: Use Nesterov momentum (default: True)

## Lion

Sign-based momentum optimizer (efficient and memory-light).

```python
optimizer = fnn.Lion(
    parameters=model.parameters(),
    lr=0.0001,
    betas=(0.95, 0.98),
    weight_decay=0.0
)
```

**Parameters:**
- `parameters`: Iterable of parameters to optimize
- `lr`: Learning rate (default: 0.0001)
- `betas`: Coefficients for computing running averages (default: (0.95, 0.98))
- `weight_decay`: Weight decay (default: 0)

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

        # Gradient clipping
        fnn.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update
        optimizer.step()

    # Print progress
    print(f"Epoch {epoch}: loss = {loss.item():.4}")
```

## Gradient Clipping

To prevent exploding gradients:

```python
# Clip gradients by global norm
fnn.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2.0)

# Clip gradients by value
fnn.clip_grad_value_(model.parameters(), clip_value=0.1)
```

## Learning Rate Scheduling

Adjust learning rate during training:

```python
# Step decay
scheduler = fnn.StepLR(optimizer, step_size=30, gamma=0.1)

# Cosine annealing
scheduler = fnn.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# Exponential decay
scheduler = fnn.ExponentialLR(optimizer, gamma=0.95)

# Reduce on plateau
scheduler = fnn.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=10,
    min_lr=1e-6,
)

# In training loop
for epoch in range(100):
    # ... training code ...
    scheduler.step()  # or scheduler.step(val_loss) for ReduceLROnPlateau
```

## Optimizer State Dict

Save and load optimizer state:

```python
# Save
fnn.save_optimizer(optimizer, 'optimizer.fno')

# Load
fnn.load_optimizer(optimizer, 'optimizer.fno')
```

## Choosing an Optimizer

| Optimizer | Use Case |
|-----------|----------|
| SGD | Simple baseline, often with momentum |
| Adam | Default choice for most tasks |
| AdamW | When weight decay is important (recommended for transformers) |
| RMSprop | Good for RNNs and non-stationary objectives |
| Muon | Orthogonalized momentum, good for large models |
| Lion | Memory-efficient, sign-based updates |
