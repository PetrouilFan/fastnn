# Models

FastNN provides pre-built model architectures.

## MLP (Multi-Layer Perceptron)

A fully-connected neural network.

```python
import fastnn as fnn

model = fnn.models.MLP(
    input_dim=784,           # Input features
    hidden_dims=[256, 128], # Hidden layer sizes
    output_dim=10,          # Output features
    activation="relu",      # Activation: "relu", "gelu", or "silu"
    dropout=0.0,            # Dropout probability (0 = no dropout)
    batch_norm=False        # Whether to use batch normalization
)

# Forward pass
output = model(input_tensor)
print(output.shape)  # (batch_size, 10)
```

### MLP with BatchNorm and Dropout

```python
model = fnn.models.MLP(
    input_dim=784,
    hidden_dims=[512, 256, 128],
    output_dim=10,
    activation="gelu",
    dropout=0.2,
    batch_norm=True
)

# Training vs. Inference
model.train()  # Enables dropout and batch norm training mode
model.eval()   # Disables dropout, uses batch norm running stats
```

## Creating Custom MLPs

Use `fnn.Sequential` directly for custom architectures:

```python
model = fnn.Sequential([
    fnn.Linear(784, 256),
    fnn.BatchNorm1d(256),
    fnn.GELU(),
    fnn.Dropout(0.3),
    fnn.Linear(256, 128),
    fnn.BatchNorm1d(128),
    fnn.GELU(),
    fnn.Dropout(0.3),
    fnn.Linear(128, 10),
])
```

## Transformer (Coming Soon)

```python
# Not yet implemented
transformer = fnn.models.Transformer(
    d_model=512,
    nhead=8,
    num_layers=6
)
# Raises: NotImplementedError
```

## Complete Training Example

```python
import fastnn as fnn

# Create data
X_train = fnn.randn(1000, 784)
y_train = fnn.randint(0, 10, (1000,))

X_test = fnn.randn(200, 784)
y_test = fnn.randint(0, 10, (200,))

# Build model
model = fnn.models.MLP(
    input_dim=784,
    hidden_dims=[256, 128],
    output_dim=10,
    activation="relu",
    dropout=0.2,
    batch_norm=True
)

# Optimizer
optimizer = fnn.Adam(model.parameters(), lr=1e-3)

# Data loaders
train_ds = fnn.TensorDataset(X_train, y_train)
test_ds = fnn.TensorDataset(X_test, y_test)

train_loader = fnn.DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = fnn.DataLoader(test_ds, batch_size=32, shuffle=False)

# Training
model.train()
for epoch in range(50):
    train_loss = 0
    for x, y in train_loader:
        pred = model(x)
        loss = fnn.cross_entropy_loss(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with fnn.no_grad():
        for x, y in test_loader:
            pred = model(x)
            preds = fnn.argmax(pred, axis=1)
            correct += (preds.numpy() == y.numpy()).sum()
            total += y.shape[0]
    
    accuracy = correct / total
    print(f"Epoch {epoch}: loss={train_loss/len(train_loader):.4}, acc={accuracy:.4}")
    
    model.train()
```
