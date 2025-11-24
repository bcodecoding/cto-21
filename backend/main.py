import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import DATA_DIR, MODEL_PRESETS, RUNS_DIR
from backend.models import (
    Dataset,
    InferenceRequest,
    InferenceResponse,
    ModelPreset,
    RunStatus,
    TrainingRequest,
    TrainingRun,
)
from backend.storage import RunStore
from backend.trainer import TrainerService

app = FastAPI(title="Trainer API")

# CORS setup
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173",
    "*",  # Allow all for demo
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

run_store = RunStore(RUNS_DIR)
trainer_service = TrainerService(run_store)


# Utility functions
def list_datasets() -> list[Dataset]:
    datasets = []
    for file in DATA_DIR.glob("**/*"):
        if file.is_file():
            datasets.append(
                Dataset(
                    id=file.stem,
                    name=file.name,
                    path=str(file),
                    size=file.stat().st_size,
                    uploaded_at=datetime.fromtimestamp(
                        file.stat().st_mtime
                    ).isoformat(),
                )
            )
    return datasets


@app.get("/models", response_model=list[ModelPreset])
def get_models():
    return [ModelPreset(**model) for model in MODEL_PRESETS]


@app.get("/datasets", response_model=list[Dataset])
def get_datasets():
    return list_datasets()


@app.post("/datasets", response_model=Dataset)
async def upload_dataset(file: UploadFile = File(...)):
    dataset_id = uuid4().hex
    dst = DATA_DIR / f"{dataset_id}_{file.filename}"
    with dst.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    dataset = Dataset(
        id=dataset_id,
        name=file.filename,
        path=str(dst),
        size=dst.stat().st_size,
        uploaded_at=datetime.utcnow().isoformat(),
    )
    return dataset


@app.post("/train", response_model=TrainingRun)
async def start_training(request: TrainingRequest):
    # Validate dataset
    datasets = list_datasets()
    dataset = next((d for d in datasets if d.id == request.dataset_id), None)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Validate model
    model = next((m for m in MODEL_PRESETS if m["id"] == request.model_id), None)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    hyperparameters = model["hyperparameters"].copy()
    if request.hyperparameters:
        hyperparameters.update(request.hyperparameters)

    run = run_store.create_run(request.model_id, request.dataset_id, hyperparameters)

    # Start training in background
    trainer_service.start_training_async(run.id)

    return run


@app.get("/runs", response_model=list[TrainingRun])
def list_runs():
    return run_store.list_runs()


@app.get("/runs/{run_id}", response_model=TrainingRun)
def get_run(run_id: str):
    run = run_store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.post("/inference", response_model=InferenceResponse)
def run_inference(request: InferenceRequest):
    run = run_store.get_run(request.run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status != RunStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Run not completed")
    # For demo, we simulate prediction
    prediction = {key: value * 0.5 for key, value in request.input_data.items()}
    return InferenceResponse(
        run_id=run.id,
        prediction=prediction,
        timestamp=datetime.utcnow().isoformat(),
    )


# Serve UI static files if built
UI_DIST = Path(__file__).resolve().parent.parent / "ui" / "dist"
if UI_DIST.exists():
    app.mount("/", StaticFiles(directory=UI_DIST, html=True), name="static")
