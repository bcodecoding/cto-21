from pydantic import BaseModel
from typing import Dict, Optional, List
from enum import Enum
from datetime import datetime


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelPreset(BaseModel):
    id: str
    name: str
    description: str
    hyperparameters: Dict[str, float]


class Dataset(BaseModel):
    id: str
    name: str
    path: str
    size: int
    uploaded_at: str


class TrainingRequest(BaseModel):
    model_id: str
    dataset_id: str
    hyperparameters: Optional[Dict[str, float]] = None


class TrainingRun(BaseModel):
    id: str
    model_id: str
    dataset_id: str
    status: RunStatus
    hyperparameters: Dict[str, float]
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    logs: List[str] = []
    metrics: Dict[str, float] = {}
    error: Optional[str] = None


class InferenceRequest(BaseModel):
    run_id: str
    input_data: Dict


class InferenceResponse(BaseModel):
    run_id: str
    prediction: Dict
    timestamp: str
