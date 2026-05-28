import fastnn as fnn

X = fnn.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], [4, 2])
y = fnn.tensor([[0.0], [1.0], [1.0], [0.0]], [4, 1])

model = fnn.models.MLP(
    input_dim=2, hidden_dims=[16, 16], output_dim=1, activation="relu"
)
optimizer = fnn.Adam(model.parameters(), lr=1e-2)

ds = fnn.TensorDataset(X, y)
loader = fnn.DataLoader(ds, batch_size=4, shuffle=True)

model.train()
for epoch in range(100):
    total_loss = 0
    for batch_x, batch_y in loader:
        pred = model(batch_x)
        loss = fnn.mse_loss(pred, batch_y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss = {total_loss / len(loader):.4}")

with fnn.no_grad():
    preds = model(X)
    print("Predictions:", preds.numpy().round(2))

print("Allocator stats:", fnn.allocator_stats())
print("Registered ops:", fnn.list_registered_ops())
