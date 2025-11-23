# ML Core Training Examples

This directory contains worked examples demonstrating how to use the `ml_core` training engine.

## train_cnn.py

A complete end-to-end training example that:
- Generates synthetic image data (horizontal vs vertical bars)
- Trains a small convolutional neural network (CNN)
- Runs for 5 epochs with validation
- Saves checkpoints, metrics, and trace logs to `artifacts/cnn_example/`

### Running the example

```bash
python examples/train_cnn.py
```

### What it demonstrates

1. **Dataset creation**: Custom PyTorch Dataset for image data
2. **Model architecture**: Simple CNN with convolutional and pooling layers
3. **Training configuration**: Epochs, batch size, learning rate, device selection
4. **Optimizer & Scheduler**: Adam optimizer with StepLR scheduler
5. **Metrics tracking**: Accuracy metric computed each epoch
6. **Checkpointing**: Model weights saved periodically
7. **Logging**: Both Python logging and JSON trace files

### Output artifacts

After running, check `artifacts/cnn_example/` for:
- `trace.jsonl`: Structured JSON event log for UI consumption
- `metrics.json`: Per-epoch metrics in JSON format
- `cnn_example_run_epoch_*.pt`: Model checkpoints

## Creating your own examples

To create a new training script:

1. Define your dataset loader (function that returns train/val DataLoaders)
2. Define your model factory (function that creates the model)
3. Define your optimizer factory
4. Optionally define a scheduler factory
5. Create a `TrainingConfig` with your hyperparameters
6. Instantiate the `Trainer` and call `train()` or `train_async()`

See `train_cnn.py` for a complete template.
