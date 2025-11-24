"""Metrics tracking and evaluation utilities."""

import json
from pathlib import Path
from typing import Any


class MetricsTracker:
    """Track and store training metrics across epochs."""

    def __init__(self):
        self._metrics: dict[str, list[float]] = {}
        self._epoch_metrics: list[dict[str, Any]] = []

    def update(self, metrics: dict[str, float], epoch: int = None):
        """
        Update metrics for the current step or epoch.

        Args:
            metrics: Dictionary of metric names and values
            epoch: Optional epoch number
        """
        for name, value in metrics.items():
            if name not in self._metrics:
                self._metrics[name] = []
            self._metrics[name].append(value)

        if epoch is not None:
            epoch_data = {"epoch": epoch, **metrics}
            self._epoch_metrics.append(epoch_data)

    def get_metric(self, name: str) -> list[float]:
        """Get all values for a specific metric."""
        return self._metrics.get(name, [])

    def get_latest(self, name: str) -> float:
        """Get the latest value for a specific metric."""
        values = self._metrics.get(name, [])
        if not values:
            raise ValueError(f"No values recorded for metric: {name}")
        return values[-1]

    def get_best(self, name: str, mode: str = "min") -> float:
        """
        Get the best value for a metric.

        Args:
            name: Metric name
            mode: 'min' or 'max'

        Returns:
            Best metric value
        """
        values = self._metrics.get(name, [])
        if not values:
            raise ValueError(f"No values recorded for metric: {name}")

        if mode == "min":
            return min(values)
        elif mode == "max":
            return max(values)
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'min' or 'max'")

    def get_all_metrics(self) -> dict[str, list[float]]:
        """Get all tracked metrics."""
        return self._metrics.copy()

    def get_epoch_metrics(self) -> list[dict[str, Any]]:
        """Get per-epoch metrics."""
        return self._epoch_metrics.copy()

    def save_json(self, filepath: str):
        """
        Save metrics to JSON file.

        Args:
            filepath: Path to save the JSON file
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {"metrics": self._metrics, "epoch_metrics": self._epoch_metrics}

        with path.open("w") as f:
            json.dump(data, f, indent=2)

    def load_json(self, filepath: str):
        """
        Load metrics from JSON file.

        Args:
            filepath: Path to the JSON file
        """
        with open(filepath) as f:
            data = json.load(f)

        self._metrics = data.get("metrics", {})
        self._epoch_metrics = data.get("epoch_metrics", [])

    def reset(self):
        """Reset all metrics."""
        self._metrics.clear()
        self._epoch_metrics.clear()

    def __repr__(self):
        return f"MetricsTracker(metrics={list(self._metrics.keys())}, epochs={len(self._epoch_metrics)})"


def compute_accuracy(predictions, targets) -> float:
    """
    Compute classification accuracy.

    Args:
        predictions: Model predictions (logits or class indices)
        targets: Ground truth labels

    Returns:
        Accuracy as a float
    """

    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=1)

    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    return correct / total if total > 0 else 0.0
