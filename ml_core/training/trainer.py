"""Core training engine implementation."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torch.utils.data import DataLoader

from ml_core.training.metrics import AccuracyMetric, Metric

DatasetLoader = Callable[
    ["TrainingConfig"],
    "DatasetBundle" | tuple[DataLoader, DataLoader | None] | DataLoader,
]
ModelFactory = Callable[["TrainingConfig"], nn.Module]
OptimizerFactory = Callable[[nn.Module, "TrainingConfig"], Optimizer]
SchedulerFactory = Callable[[Optimizer, "TrainingConfig"], _LRScheduler]
LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class TrainingConfig:
    """Configuration for a training run."""

    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 1e-3
    device: str = "auto"
    checkpoint_interval: int = 1
    num_workers: int = 0
    run_name: str = field(
        default_factory=lambda: datetime.utcnow().strftime("run-%Y%m%d-%H%M%S")
    )
    artifact_dir: Path | None = None
    trace_filename: str = "trace.jsonl"
    metrics_filename: str = "metrics.json"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["artifact_dir"] = (
            str(data["artifact_dir"]) if data["artifact_dir"] else None
        )
        return data


@dataclass
class DatasetBundle:
    """Container for train/validation loaders."""

    train: DataLoader
    val: DataLoader | None = None


@dataclass
class EpochResult:
    epoch: int
    train_metrics: dict[str, float]
    val_metrics: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "train": self.train_metrics,
            "val": self.val_metrics,
        }


@dataclass
class TrainingResult:
    config: TrainingConfig
    history: list[EpochResult]
    checkpoints: list[Path]
    artifact_dir: Path
    metrics_path: Path
    trace_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "history": [epoch.to_dict() for epoch in self.history],
            "checkpoints": [str(path) for path in self.checkpoints],
            "artifact_dir": str(self.artifact_dir),
            "metrics_path": str(self.metrics_path),
            "trace_path": str(self.trace_path),
        }


class TraceLogger:
    """Write JSON trace lines for UI consumption."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log_event(self, event_type: str, **payload: Any) -> None:
        entry = {
            "event": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            **payload,
        }
        serialized = json.dumps(entry, default=_json_default)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(serialized + "\n")


class Trainer:
    """High level training service used by the backend/API."""

    def __init__(
        self,
        config: TrainingConfig,
        dataset_loader: DatasetLoader,
        model_factory: ModelFactory,
        optimizer_factory: OptimizerFactory,
        loss_fn: LossFunction | nn.Module,
        scheduler_factory: SchedulerFactory | None = None,
        metrics: dict[str, Metric] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.dataset_loader = dataset_loader
        self.model_factory = model_factory
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.loss_fn = loss_fn
        self.metrics = metrics or {"accuracy": AccuracyMetric()}
        self.logger = logger or logging.getLogger("ml_core.training")
        self.logger.setLevel(logging.INFO)

        self.device = self._resolve_device(config.device)
        self.artifact_dir = self._resolve_artifact_dir()
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.artifact_dir / self.config.trace_filename
        self.metrics_path = self.artifact_dir / self.config.metrics_filename
        self.trace_logger = TraceLogger(self.trace_path)

        if hasattr(self.loss_fn, "to"):
            self.loss_fn = self.loss_fn.to(self.device)  # type: ignore[assignment]

    def train(self) -> TrainingResult:
        """Run training synchronously."""
        return self._train_impl()

    async def train_async(self) -> TrainingResult:
        """Run training asynchronously in the background."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._train_impl)

    def train_in_background(self) -> asyncio.Task[TrainingResult]:
        """Convenience helper to start training as an asyncio task."""
        return asyncio.create_task(self.train_async())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _train_impl(self) -> TrainingResult:
        dataset_bundle = self._resolve_dataset()
        model = self.model_factory(self.config).to(self.device)
        optimizer = self.optimizer_factory(model, self.config)
        scheduler = (
            self.scheduler_factory(optimizer, self.config)
            if self.scheduler_factory
            else None
        )

        history: list[EpochResult] = []
        checkpoints: list[Path] = []

        self.logger.info("Starting training run %s", self.config.run_name)
        self.trace_logger.log_event(
            "training_started", config=self.config.to_dict(), device=str(self.device)
        )

        for epoch in range(1, self.config.epochs + 1):
            self.logger.info("Epoch %s/%s", epoch, self.config.epochs)
            train_metrics = self._run_epoch(
                model=model,
                dataloader=dataset_bundle.train,
                optimizer=optimizer,
                train=True,
            )
            self.trace_logger.log_event(
                "epoch_complete", epoch=epoch, split="train", metrics=train_metrics
            )

            val_metrics = None
            if dataset_bundle.val is not None:
                val_metrics = self._run_epoch(
                    model=model,
                    dataloader=dataset_bundle.val,
                    optimizer=None,
                    train=False,
                )
                self.trace_logger.log_event(
                    "epoch_complete", epoch=epoch, split="val", metrics=val_metrics
                )

            history.append(
                EpochResult(
                    epoch=epoch, train_metrics=train_metrics, val_metrics=val_metrics
                )
            )

            if scheduler is not None:
                self._step_scheduler(scheduler, train_metrics, val_metrics)

            if (
                epoch % self.config.checkpoint_interval == 0
                or epoch == self.config.epochs
            ):
                checkpoint_path = self._save_checkpoint(model, optimizer, epoch)
                checkpoints.append(checkpoint_path)

        self._write_history(history)
        self.trace_logger.log_event(
            "training_completed",
            epochs=self.config.epochs,
            metrics_file=str(self.metrics_path),
            checkpoints=[str(path) for path in checkpoints],
        )

        return TrainingResult(
            config=self.config,
            history=history,
            checkpoints=checkpoints,
            artifact_dir=self.artifact_dir,
            metrics_path=self.metrics_path,
            trace_path=self.trace_path,
        )

    def _run_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: Optimizer | None,
        train: bool,
    ) -> dict[str, float]:
        if dataloader is None:
            raise ValueError("DataLoader is required for training")

        metric_instances = self._clone_metrics()
        running_loss = 0.0
        total_samples = 0

        if train:
            model.train()
            context = contextlib.nullcontext()
        else:
            model.eval()
            context = torch.no_grad()

        with context:
            for inputs, targets in self._iterate_batches(dataloader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss_tensor = (
                    loss
                    if isinstance(loss, torch.Tensor)
                    else torch.tensor(loss, device=self.device)
                )

                if train:
                    if optimizer is None:
                        raise ValueError("Optimizer must be provided for training")
                    optimizer.zero_grad()
                    loss_tensor.backward()
                    optimizer.step()

                batch_size = targets.size(0)
                running_loss += loss_tensor.detach().item() * batch_size
                total_samples += batch_size

                for metric in metric_instances.values():
                    metric.update(outputs.detach(), targets)

        metrics = {
            name: float(metric.compute()) for name, metric in metric_instances.items()
        }
        metrics["loss"] = running_loss / max(1, total_samples)

        if train:
            self.logger.info(
                "Training - loss: %.4f %s",
                metrics["loss"],
                " ".join(
                    f"{name}:{value:.4f}"
                    for name, value in metrics.items()
                    if name != "loss"
                ),
            )
        else:
            self.logger.info(
                "Validation - loss: %.4f %s",
                metrics["loss"],
                " ".join(
                    f"{name}:{value:.4f}"
                    for name, value in metrics.items()
                    if name != "loss"
                ),
            )

        return metrics

    def _iterate_batches(
        self, dataloader: DataLoader
    ) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        for batch in dataloader:
            if isinstance(batch, dict):
                inputs = batch["inputs"]
                targets = batch["targets"]
            elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                inputs, targets = batch[0], batch[1]
            else:  # pragma: no cover - defensive branch
                raise ValueError("Unsupported batch format")
            yield inputs, targets

    def _clone_metrics(self) -> dict[str, Metric]:
        return {name: metric.clone() for name, metric in self.metrics.items()}

    def _step_scheduler(
        self,
        scheduler: _LRScheduler,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ) -> None:
        if isinstance(scheduler, ReduceLROnPlateau):
            monitor = (val_metrics or train_metrics).get("loss")
            if monitor is None:
                raise ValueError("ReduceLROnPlateau requires a loss metric to monitor")
            scheduler.step(monitor)
        else:
            scheduler.step()

        lr = scheduler.optimizer.param_groups[0]["lr"]
        self.trace_logger.log_event("lr_updated", lr=lr)

    def _save_checkpoint(
        self, model: nn.Module, optimizer: Optimizer, epoch: int
    ) -> Path:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": self.config.to_dict(),
        }
        path = self.artifact_dir / f"{self.config.run_name}_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        self.trace_logger.log_event("checkpoint_saved", epoch=epoch, path=str(path))
        return path

    def _write_history(self, history: list[EpochResult]) -> None:
        payload = [epoch.to_dict() for epoch in history]
        with self.metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _resolve_dataset(self) -> DatasetBundle:
        loaders = self.dataset_loader(self.config)
        if isinstance(loaders, DatasetBundle):
            bundle = loaders
        elif isinstance(loaders, DataLoader):
            bundle = DatasetBundle(train=loaders)
        else:
            train_loader, val_loader = loaders
            bundle = DatasetBundle(train=train_loader, val=val_loader)

        if bundle.train is None:
            raise ValueError("Dataset loader must provide a training DataLoader")
        return bundle

    def _resolve_artifact_dir(self) -> Path:
        if self.config.artifact_dir:
            return Path(self.config.artifact_dir)
        return Path("artifacts") / self.config.run_name

    def _resolve_device(self, device_name: str) -> torch.device:
        if device_name == "auto":
            if torch.cuda.is_available():
                target = "cuda"
            elif (
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ):
                target = "mps"
            else:
                target = "cpu"
        else:
            target = device_name
        try:
            return torch.device(target)
        except RuntimeError as exc:  # pragma: no cover - hardware dependent
            raise ValueError(f"Invalid device '{device_name}': {exc}") from exc


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = [
    "Trainer",
    "TrainingConfig",
    "TrainingResult",
    "EpochResult",
    "DatasetBundle",
]
