import sys

sys.path.append("target/debug")
import fastnn

# Create a simple model
model = fastnn.models.Transformer(
    vocab_size=100,
    max_seq_len=16,
    d_model=64,
    num_heads=4,
    num_layers=2,
    ff_dim=128,
    num_classes=2,
    dropout_p=0.1,
)

params = model.parameters()
print(f"Total params: {len(params)}")
for i, p in enumerate(params):
    print(f"Param {i}: shape {p.shape}")
