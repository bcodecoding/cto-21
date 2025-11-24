from enum import Enum

from pydantic import BaseModel


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelPreset(BaseModel):
    id: str
    name: str
    description: str
    hyperparameters: dict[str, float]


class Dataset(BaseModel):
    id: str
    name: str
    path: str
    size: int
    uploaded_at: str


class TrainingRequest(BaseModel):
    model_id: str
    dataset_id: str
    hyperparameters: dict[str, float] | None = None


class TrainingRun(BaseModel):
    id: str
    model_id: str
    dataset_id: str
    status: RunStatus
    hyperparameters: dict[str, float]
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    logs: list[str] = []
    metrics: dict[str, float] = {}
    error: str | None = None


class InferenceRequest(BaseModel):
    run_id: str
    input_data: dict


class InferenceResponse(BaseModel):
    run_id: str
    prediction: dict
    timestamp: str
