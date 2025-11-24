# ML Training Platform

A full-stack machine learning training platform with FastAPI backend and React frontend. This platform allows you to configure training runs, manage datasets, start training jobs, and monitor progress in real-time.

## Quick Start

### Prerequisites

- Python 3.11+ 
- Node.js 20+ and npm
- [uv](https://docs.astral.sh/uv/) (recommended Python package manager)

### Installation

1. Clone the repository and navigate to the project directory
2. Install dependencies and setup the development environment:

```bash
make install
```

This will:
- Install Python dependencies using `uv`
- Install Node.js dependencies for the frontend
- Set up the development environment

### Development

Start both backend and frontend servers:

```bash
make dev
```

Or start them individually:

```bash
make api    # Start FastAPI backend on http://localhost:8000
make ui     # Start React frontend on http://localhost:5173
```

### Development Commands

- `make install` - Install all dependencies
- `make dev` - Start both development servers
- `make api` - Start FastAPI backend only
- `make ui` - Start React frontend only
- `make lint` - Run linting checks
- `make format` - Format code with black and ruff
- `make test` - Run tests
- `make clean` - Clean cache and build files
- `make check` - Run all checks (lint + test)

### Code Quality

The project uses:
- **Black** for code formatting
- **Ruff** for linting and import sorting
- **pytest** for testing
- **pre-commit** for git hooks (optional)

## Features

- **Model Presets**: Choose from pre-configured model architectures (ResNet-50, BERT Base, Custom)
- **Dataset Management**: Upload and manage training datasets
- **Training Jobs**: Start training jobs with customizable hyperparameters
- **Real-time Monitoring**: Track training progress with live logs and metrics
- **RESTful API**: Complete REST API for programmatic access
- **CORS Support**: Fully configured CORS for frontend-backend communication

## Project Structure

```
.
├── backend/          # FastAPI backend
│   ├── main.py       # Main application with API endpoints
│   ├── models.py     # Pydantic models
│   ├── storage.py    # Run metadata storage
│   ├── trainer.py    # Training service
│   └── config.py     # Configuration
├── data_pipeline/    # Modular data loading & preprocessing
│   ├── __init__.py
│   ├── base.py       # Base dataset and transform classes
│   ├── configs.py    # Configuration objects (JSON/YAML compatible)
│   ├── transforms.py # Image, text, tabular transforms
│   ├── registries.py # Dataset registry
│   ├── loaders/
│   │   ├── image_loader.py      # Image folder datasets
│   │   ├── text_loader.py       # Text (JSONL/CSV) datasets
│   │   └── tabular_loader.py    # Tabular CSV datasets
│   └── README.md     # Data pipeline documentation
├── ml_core/          # ML training engine
│   ├── datasets/     # Dataset utilities
│   ├── models/       # Model factories
│   ├── training/     # Trainer and metrics
│   └── utils/        # Utility functions
├── ui/               # React frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── ModelSelector.jsx
│   │   │   ├── DatasetUploader.jsx
│   │   │   └── TrainingMonitor.jsx
│   │   ├── App.jsx
│   │   ├── api.js
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
├── examples/         # Example scripts and datasets
│   ├── data_pipeline_demo.py    # Data pipeline usage examples
│   ├── DATA_PIPELINE.md         # Data pipeline examples guide
│   ├── train_cnn.py             # CNN training example
│   ├── README.md                # Examples documentation
│   └── data/                    # Sample datasets
│       ├── images/              # Sample images (cat/dog)
│       ├── text_samples.jsonl   # Sample text data
│       ├── tabular_sample.csv   # Sample tabular data
│       └── configs/             # Sample configurations
├── data/             # Dataset storage
├── runs/             # Training run metadata
├── tests/            # Unit and integration tests
│   └── test_data_pipeline.py    # Data pipeline tests
└── requirements.txt
```

## Prerequisites

- Python 3.8+
- Node.js 16+ and npm
- pip

## Installation

### Backend Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

### Frontend Setup

1. Navigate to the UI directory and install dependencies:
```bash
cd ui
npm install
cd ..
```

## Usage

### Option 1: Development Mode (Separate Frontend & Backend)

This mode is best for development with hot-reloading.

#### Start the Backend

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

API Documentation (Swagger UI): `http://localhost:8000/docs`

#### Start the Frontend

In a separate terminal:

```bash
cd ui
npm run dev
```

The UI will be available at `http://localhost:5173`

### Option 2: Production Mode (Bundled Static Build)

Build the frontend and serve it through FastAPI.

#### Build the Frontend

```bash
cd ui
npm run build
cd ..
```

#### Start the Backend

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

The complete application (API + UI) will be available at `http://localhost:8000`

## Running a Sample Training Job

### Via the Web UI

1. Start the backend and frontend as described above
2. Open the UI in your browser (`http://localhost:5173` in dev mode or `http://localhost:8000` in production mode)
3. Select a model preset (e.g., "ResNet-50")
4. Select the sample dataset (sample_data.csv is included)
5. Click "Start Training"
6. Monitor the training progress in real-time with logs and metrics

### Via the API

You can also interact with the API directly using curl or any HTTP client:

#### Get Available Models
```bash
curl http://localhost:8000/models
```

#### Get Available Datasets
```bash
curl http://localhost:8000/datasets
```

#### Start Training
```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "resnet50",
    "dataset_id": "sample_data"
  }'
```

#### Check Run Status
```bash
curl http://localhost:8000/runs/{run_id}
```

#### List All Runs
```bash
curl http://localhost:8000/runs
```

## API Endpoints

### Models

- `GET /models` - List all available model presets

### Datasets

- `GET /datasets` - List all datasets
- `POST /datasets` - Upload a new dataset (multipart/form-data)

### Training

- `POST /train` - Start a training job
  ```json
  {
    "model_id": "resnet50",
    "dataset_id": "sample_data",
    "hyperparameters": {
      "learning_rate": 0.001,
      "epochs": 3,
      "batch_size": 32
    }
  }
  ```

### Runs

- `GET /runs` - List all training runs
- `GET /runs/{id}` - Get details of a specific run

### Inference

- `POST /inference` - Run inference on a completed training run
  ```json
  {
    "run_id": "run-uuid",
    "input_data": {"feature1": 1.0, "feature2": 0.5}
  }
  ```

## Data Storage

- **Training Runs**: Stored as JSON files in the `runs/` directory
- **Datasets**: Stored in the `data/` directory
- **Sample Data**: A sample CSV dataset is included at `data/sample_data.csv`

## CORS Configuration

The backend is configured to accept requests from:
- `http://localhost:3000`
- `http://localhost:5173`
- All origins (for demo purposes)

Modify the `origins` list in `backend/main.py` for production deployments.

## Training Simulation

For demonstration purposes, the training service simulates a training process:
- Takes ~2 seconds per epoch
- Generates simulated loss and accuracy metrics
- Updates logs in real-time
- Completes after the specified number of epochs

In a production environment, you would replace `backend/trainer.py` with actual ML training logic using PyTorch, TensorFlow, or other frameworks.

## ML Core Training Engine

The repository also ships with a reusable training engine located at `ml_core/training`. The `Trainer` class wires together dataset loaders, model factories, optimizers, schedulers, checkpointing, and metric tracking. It emits structured JSON traces for UI consumption and can be run synchronously or scheduled as an asyncio background task.

### Running the worked example

A complete example that trains a small CNN on synthetic image data lives at `examples/train_cnn.py`. Running the script will execute the end-to-end training loop and save checkpoints + metrics under `artifacts/cnn_example`:

```bash
python examples/train_cnn.py
```

Review the generated trace (`trace.jsonl`) and `metrics.json` files inside the artifact directory to inspect how the trainer reports progress.

## Data Pipeline Module

The repository includes a modular data loading and preprocessing layer (`data_pipeline/`) that provides abstractions for dataset management and concrete loaders for multiple data modalities.

### Features

- **Image Dataset Loader**: Load images from folder structures with automatic resizing and normalization
- **Text Dataset Loader**: Load text from JSONL/CSV files with tokenization
- **Tabular Dataset Loader**: Load CSV files with automatic numeric scaling
- **Configuration-Based Instantiation**: Define datasets via JSON configuration files
- **Registry System**: Central management of datasets and loaders
- **PyTorch Integration**: Full `Dataset`/`DataLoader` compatibility

### Quick Start

```python
from data_pipeline import ImageDatasetConfig, create_default_registry

# Define configuration
config = ImageDatasetConfig(
    name="my_images",
    path="path/to/images",
    image_size=(224, 224),
    normalize=True,
)

# Load dataset
registry = create_default_registry()
dataset = registry.load_from_config(config)
loader = registry.create_dataloader(dataset, batch_size=32)

# Iterate batches
for images, labels in loader:
    # images: (B, 3, 224, 224)
    # labels: (B,)
    pass
```

### Running the Demo

See all dataset types in action:

```bash
python examples/data_pipeline_demo.py
```

This runs demonstrations of:
1. Image dataset loading with transforms
2. Text dataset loading with tokenization
3. Tabular dataset loading with numeric scaling
4. Configuration-based dataset instantiation
5. PyTorch DataLoader integration

### Sample Datasets

Located in `examples/data/`:

- **Images**: `images/` - Sample cat and dog images (2 classes, 6 images total)
- **Text**: `text_samples.jsonl` - Sentiment examples (3 classes, 9 samples)
- **Tabular**: `tabular_sample.csv` - Loan approval data (2 classes, 12 samples)
- **Configs**: `configs/` - JSON configuration examples for all dataset types

### Documentation

- **[Data Pipeline README](data_pipeline/README.md)** - Complete API reference and guide
- **[Examples Guide](examples/DATA_PIPELINE.md)** - Usage examples and custom dataset creation
- **[Tests](tests/test_data_pipeline.py)** - Comprehensive unit and integration tests

### Supported Formats

**Images**: `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.ppm`
- Expected structure: `root_dir/class1/img1.jpg`, `root_dir/class2/img2.jpg`

**Text**: JSONL or CSV
- JSONL: `{"prompt": "text", "label": "class"}`
- CSV: `prompt,label` header with rows

**Tabular**: CSV with headers
- `feature1,feature2,...,label`

## Development

### Adding New Model Presets

Edit `backend/config.py` and add new model configurations to the `MODEL_PRESETS` list:

```python
{
    "id": "my-model",
    "name": "My Model",
    "description": "Description",
    "hyperparameters": {
        "learning_rate": 0.001,
        "epochs": 5,
        "batch_size": 32,
    },
}
```

### Extending the Trainer

Replace the simulated training in `backend/trainer.py` with actual ML training code:

```python
async def train_model(self, run_id: str):
    run = self.run_store.get_run(run_id)
    
    # Load your actual model and dataset
    model = load_model(run.model_id)
    dataset = load_dataset(run.dataset_id)
    
    # Train with real ML framework
    for epoch in range(run.hyperparameters['epochs']):
        # Your training loop
        loss, accuracy = train_epoch(model, dataset)
        
        # Update metrics
        self.run_store.update_metrics(run_id, {
            f"epoch_{epoch}_loss": loss,
            f"epoch_{epoch}_accuracy": accuracy,
        })
```

## Troubleshooting

### Port Already in Use

If port 8000 or 5173 is already in use, specify a different port:

```bash
# Backend
uvicorn backend.main:app --port 8001

# Frontend
cd ui
npm run dev -- --port 5174
```

### CORS Errors

Ensure the backend is running before starting the frontend, and check that the API_BASE URL in `ui/src/api.js` matches your backend URL.

### Dependencies Not Found

Make sure you've installed all dependencies:
```bash
pip install -r requirements.txt
cd ui && npm install
```

## License

MIT
