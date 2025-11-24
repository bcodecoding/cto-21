"""Common metric implementations for the training engine."""

from __future__ import annotations

from typing import Protocol

import torch


class Metric(Protocol):
    """Interface for all metrics used by the trainer."""

    def reset(self) -> None: ...

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None: ...

    def compute(self) -> float: ...

    def clone(self) -> Metric:  # pragma: no cover - interface method
        ...


class AccuracyMetric:
    """Simple classification accuracy metric."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._correct = 0
        self._total = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        if predictions.dim() > 1:
            predicted = predictions.argmax(dim=1)
        else:
            predicted = predictions
        self._correct += (predicted == targets).sum().item()
        self._total += targets.numel()

    def compute(self) -> float:
        if self._total == 0:
            return 0.0
        return self._correct / self._total

    def clone(self) -> AccuracyMetric:
        return AccuracyMetric()


__all__ = ["Metric", "AccuracyMetric"]
