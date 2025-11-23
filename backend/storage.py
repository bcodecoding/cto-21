import json
import uuid
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

from backend.models import TrainingRun, RunStatus


class RunStore:
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(exist_ok=True)

    def _run_file(self, run_id: str) -> Path:
        return self.storage_path / f"{run_id}.json"

    def list_runs(self) -> List[TrainingRun]:
        runs = []
        for file in self.storage_path.glob("*.json"):
            try:
                data = json.loads(file.read_text())
                runs.append(TrainingRun(**data))
            except Exception:
                continue
        return sorted(runs, key=lambda run: run.created_at, reverse=True)

    def get_run(self, run_id: str) -> Optional[TrainingRun]:
        file = self._run_file(run_id)
        if not file.exists():
            return None
        data = json.loads(file.read_text())
        return TrainingRun(**data)

    def save_run(self, run: TrainingRun) -> TrainingRun:
        file = self._run_file(run.id)
        file.write_text(run.json())
        return run

    def create_run(
        self, model_id: str, dataset_id: str, hyperparameters: Dict[str, float]
    ) -> TrainingRun:
        run = TrainingRun(
            id=str(uuid.uuid4()),
            model_id=model_id,
            dataset_id=dataset_id,
            status=RunStatus.PENDING,
            hyperparameters=hyperparameters,
            created_at=datetime.utcnow().isoformat(),
            logs=[],
            metrics={},
        )
        return self.save_run(run)

    def update_status(
        self, run_id: str, status: RunStatus, error: Optional[str] = None
    ) -> Optional[TrainingRun]:
        run = self.get_run(run_id)
        if not run:
            return None
        run.status = status
        timestamp = datetime.utcnow().isoformat()
        if status == RunStatus.RUNNING:
            run.started_at = timestamp
        elif status in (RunStatus.COMPLETED, RunStatus.FAILED):
            run.completed_at = timestamp
        run.error = error
        return self.save_run(run)

    def append_log(self, run_id: str, message: str) -> Optional[TrainingRun]:
        run = self.get_run(run_id)
        if not run:
            return None
        run.logs.append(f"[{datetime.utcnow().isoformat()}] {message}")
        return self.save_run(run)

    def update_metrics(self, run_id: str, metrics: Dict[str, float]) -> Optional[TrainingRun]:
        run = self.get_run(run_id)
        if not run:
            return None
        run.metrics.update(metrics)
        return self.save_run(run)
