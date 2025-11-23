from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

DATA_DIR.mkdir(exist_ok=True)
RUNS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Sample presets for models
MODEL_PRESETS = [
    {
        "id": "resnet50",
        "name": "ResNet-50",
        "description": "Image classification model preset",
        "hyperparameters": {
            "learning_rate": 0.001,
            "epochs": 3,
            "batch_size": 32,
        },
    },
    {
        "id": "bert-base",
        "name": "BERT Base",
        "description": "Transformer model for NLP",
        "hyperparameters": {
            "learning_rate": 2e-5,
            "epochs": 2,
            "batch_size": 16,
        },
    },
    {
        "id": "custom",
        "name": "Custom Model",
        "description": "Customizable model template",
        "hyperparameters": {
            "learning_rate": 0.0005,
            "epochs": 5,
            "batch_size": 64,
        },
    },
]
