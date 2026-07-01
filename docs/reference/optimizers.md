# Optimizers

Optimization algorithms for training neural networks, all accessible from `import fastnn as fnn`.

## SGD

Stochastic gradient descent with optional momentum and Nesterov acceleration.

```python
optimizer = fnn.SGD(
    parameters=model.parameters(),
    lr=0.01,              # Learning rate (required)
    momentum=0.0,         # Momentum factor
    weight_decay=0.0,     # L2 regularization
    nesterov=False        # Use Nesterov momentum
)
```

## Adam

Adaptive Moment Estimation.

```python
optimizer = fnn.Adam(
    parameters=model.parameters(),
    lr=0.001,               # Learning rate
    betas=(0.9, 0.999),     # Coefficients for running averages
    eps=1e-8,               # Numerical stability term
    weight_decay=0.0        # L2 regularization
)
```

## AdamW

Adam with decoupled weight decay. Applies weight decay directly to parameters rather than incorporating it into the gradient update.

```python
optimizer = fnn.AdamW(
    parameters=model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01   # Decoupled weight decay
)
```

## RMSprop

Root mean square propagation.

```python
optimizer = fnn.RMSprop(
    parameters=model.parameters(),
    lr=0.01,
    alpha=0.99,           # Smoothing constant for squared gradient
    eps=1e-8,
    weight_decay=0.0,
    momentum=0.0,
    centered=False        # Centered RMSprop variant
)
```

## Muon

Orthogonalized momentum optimizer with Nesterov acceleration.

```python
optimizer = fnn.Muon(
    parameters=model.parameters(),
    lr=0.025,
    momentum=0.95,
    weight_decay=0.0,
    nesterov=True
)
```

## Lion

Sign-based momentum optimizer. Memory-light alternative using sign operations instead of full adaptive updates.

```python
optimizer = fnn.Lion(
    parameters=model.parameters(),
    lr=0.0001,
    betas=(0.95, 0.98),
    weight_decay=0.0
)
```

## Training Loop Pattern

```python
import fastnn as fnn

model = fnn.models.MLP(input_dim=784, hidden_dims=[256, 128], output_dim=10)
optimizer = fnn.Adam(model.parameters(), lr=1e-3)
loss_fn = fnn.mse_loss

model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        x, y = batch
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        fnn.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
```

## Learning Rate Schedulers

```python
# Step decay
scheduler = fnn.StepLR(optimizer, step_size=30, gamma=0.1)

# Cosine annealing
scheduler = fnn.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# Exponential decay
scheduler = fnn.ExponentialLR(optimizer, gamma=0.95)

# Reduce on plateau
scheduler = fnn.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-6,
)

# In training loop
for epoch in range(100):
    # training code ...
    scheduler.step()  # or scheduler.step(val_loss) for ReduceLROnPlateau
```

## Optimizer State

```python
state = optimizer.state_dict()        # Save
optimizer.load_state_dict(state)      # Restore
```

## Quick Reference

| Optimizer | Use Case |
|-----------|----------|
| SGD | Simple baseline with momentum |
| Adam | Default choice for most tasks |
| AdamW | Transformers and when weight decay matters |
| RMSprop | RNNs and non-stationary objectives |
| Muon | Large models with orthogonalized momentum |
| Lion | Memory-efficient sign-based updates |

## See also

- [Neural Network Modules](nn-modules.md) -- Layers to optimize
- [Tensors](tensors.md) -- Gradient computation and clipping
- [API Reference](api-reference.md) -- Full optimizer signatures
- [Python API](python-api.md) -- Compiled training optimizer strings
- [Training Basics](../guides/training/training-basics.md) -- Training loop walkthrough
- [Index](../index.md) -- Full documentation index
