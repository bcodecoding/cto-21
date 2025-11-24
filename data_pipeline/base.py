"""Base classes for dataset abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """Abstract base class for all dataset implementations."""

    def __init__(self, name: str, description: str = "") -> None:
        """Initialize base dataset.

        Args:
            name: Dataset name
            description: Optional dataset description
        """
        self.name = name
        self.description = description
        self._samples: list[Any] = []

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """Get a single sample by index."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, len={len(self)})"


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders."""

    @abstractmethod
    def load(self, config: Any) -> BaseDataset:
        """Load a dataset from a configuration.

        Args:
            config: Dataset configuration object

        Returns:
            Loaded dataset instance
        """
        pass


class Transform(ABC):
    """Abstract base class for data transforms."""

    @abstractmethod
    def __call__(self, sample: Any) -> Any:
        """Apply the transform to a sample."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Compose:
    """Compose multiple transforms sequentially."""

    def __init__(self, transforms: list[Transform]) -> None:
        """Initialize with list of transforms.

        Args:
            transforms: List of Transform instances
        """
        self.transforms = transforms

    def __call__(self, sample: Any) -> Any:
        """Apply all transforms in sequence."""
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def __repr__(self) -> str:
        lines = ["Compose("]
        for t in self.transforms:
            lines.append(f"    {t!r},")
        lines.append(")")
        return "\n".join(lines)
