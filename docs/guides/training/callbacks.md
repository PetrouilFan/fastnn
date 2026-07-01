# Callbacks & Checkpointing

Callbacks provide hooks into the training loop for logging, checkpointing, early stopping, and metrics tracking.

## Using Callbacks

```python
import fastnn as fnn

callbacks = [
    fnn.callbacks.ModelCheckpoint(
        filepath="checkpoint_{epoch}.fnn",
        monitor="val_loss",
        save_best_only=True,
    ),
    fnn.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
    ),
    fnn.callbacks.LearningRateScheduler(
        schedule=lambda epoch: 0.001 * (0.95 ** epoch),
    ),
]

model.fit(
    train_loader,
    val_loader=val_loader,
    epochs=100,
    callbacks=callbacks,
)
```

## Built-in Callbacks

### ModelCheckpoint

```python
fnn.callbacks.ModelCheckpoint(
    filepath="model_{epoch}.fnn",
    monitor="val_loss",
    save_best_only=True,
    mode="min",            # "min" or "max"
    save_weights_only=False,
)
```

Saves model checkpoints during training. When `save_best_only=True`, only saves when the monitored metric improves.

### EarlyStopping

```python
fnn.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    min_delta=0.001,
    mode="min",
    restore_best_weights=True,
)
```

Stops training when a monitored metric stops improving.

### LearningRateScheduler

```python
fnn.callbacks.LearningRateScheduler(
    schedule=lambda epoch: 0.001 * (0.95 ** epoch),
    verbose=True,
)
```

Adjusts the learning rate after each epoch.

### MetricsLogger

```python
fnn.callbacks.MetricsLogger(
    log_file="training.log",
    metrics=["loss", "val_loss", "accuracy"],
)
```

Logs training metrics to a file or stdout.

### TensorBoard-like Logging

```python
fnn.callbacks.CsvLogger(
    filename="training_log.csv",
    separator=",",
    append=False,
)
```

Logs metrics to a CSV file for post-hoc analysis.

## Custom Callbacks

```python
class MyCallback(fnn.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"Starting epoch {epoch}")

    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch}: loss = {logs['loss']:.4f}")

    def on_train_begin(self, logs=None):
        print("Training started")

    def on_train_end(self, logs=None):
        print("Training finished")
```

## See Also

- [Training Basics](training-basics.md) — training loop fundamentals
- [Distributed Training](distributed.md) — multi-device training
