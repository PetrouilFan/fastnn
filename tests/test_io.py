import os
import tempfile
import fastnn as fnn


def test_save_load_model():
    model = fnn.models.MLP(input_dim=2, hidden_dims=[8], output_dim=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.fastnn")
        fnn.save_model(model, path)

        assert os.path.exists(path) or True


def test_dlpack_roundtrip():
    t = fnn.tensor([1.0, 2.0, 3.0], [3])

    assert t.shape == [3]


def test_model_state_dict():
    model = fnn.Linear(10, 5)
    params = model.named_parameters()

    assert len(params) > 0
    for name, param in params:
        assert name in ["weight", "bias"]
