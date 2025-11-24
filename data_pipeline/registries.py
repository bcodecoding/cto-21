"""Dataset registry for managing and instantiating datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from data_pipeline.base import BaseDataset
from data_pipeline.configs import (
    DatasetConfig,
    ImageDatasetConfig,
    TabularDatasetConfig,
    TextDatasetConfig,
)


class DatasetRegistry:
    """Central registry for dataset management."""

    def __init__(self) -> None:
        """Initialize registry."""
        self._datasets: Dict[str, BaseDataset] = {}
        self._configs: Dict[str, DatasetConfig] = {}
        self._loaders: Dict[str, Callable] = {}

    def register_dataset(
        self,
        name: str,
        dataset: BaseDataset,
        config: Optional[DatasetConfig] = None,
    ) -> None:
        """Register a dataset.

        Args:
            name: Dataset name
            dataset: BaseDataset instance
            config: Optional configuration
        """
        self._datasets[name] = dataset
        if config:
            self._configs[name] = config

    def register_loader(
        self,
        dataset_type: str,
        loader_fn: Callable[[DatasetConfig], BaseDataset],
    ) -> None:
        """Register a custom loader function.

        Args:
            dataset_type: Type identifier (e.g., 'image', 'text', 'tabular')
            loader_fn: Callable that creates a dataset from config
        """
        self._loaders[dataset_type] = loader_fn

    def get_dataset(self, name: str) -> Optional[BaseDataset]:
        """Get registered dataset by name.

        Args:
            name: Dataset name

        Returns:
            Dataset or None if not found
        """
        return self._datasets.get(name)

    def get_config(self, name: str) -> Optional[DatasetConfig]:
        """Get configuration for a dataset.

        Args:
            name: Dataset name

        Returns:
            Configuration or None if not found
        """
        return self._configs.get(name)

    def list_datasets(self) -> list[str]:
        """List all registered dataset names.

        Returns:
            List of dataset names
        """
        return list(self._datasets.keys())

    def load_from_config(self, config: DatasetConfig) -> BaseDataset:
        """Load dataset from configuration.

        Args:
            config: Dataset configuration

        Returns:
            Loaded dataset

        Raises:
            ValueError: If dataset type is not supported
        """
        dataset_type = config.dataset_type

        if dataset_type not in self._loaders:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

        loader = self._loaders[dataset_type]
        return loader(config)

    def create_dataloader(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        """Create a PyTorch DataLoader from a dataset.

        Args:
            dataset: Dataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes

        Returns:
            DataLoader instance
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def remove_dataset(self, name: str) -> bool:
        """Remove a registered dataset.

        Args:
            name: Dataset name

        Returns:
            True if removed, False if not found
        """
        if name in self._datasets:
            del self._datasets[name]
            if name in self._configs:
                del self._configs[name]
            return True
        return False

    def clear(self) -> None:
        """Clear all registered datasets and configs."""
        self._datasets.clear()
        self._configs.clear()

    def __repr__(self) -> str:
        return f"DatasetRegistry(datasets={list(self._datasets.keys())}, types={list(self._loaders.keys())})"


def create_default_registry() -> DatasetRegistry:
    """Create a registry with default loaders.

    Returns:
        Configured DatasetRegistry
    """
    from data_pipeline.loaders.image_loader import ImageDataset
    from data_pipeline.loaders.tabular_loader import TabularDataset
    from data_pipeline.loaders.text_loader import TextDataset

    registry = DatasetRegistry()

    def load_image_dataset(config: ImageDatasetConfig) -> BaseDataset:
        return ImageDataset.from_config(config)

    def load_text_dataset(config: TextDatasetConfig) -> BaseDataset:
        return TextDataset.from_config(config)

    def load_tabular_dataset(config: TabularDatasetConfig) -> BaseDataset:
        return TabularDataset.from_config(config)

    registry.register_loader("image", load_image_dataset)
    registry.register_loader("text", load_text_dataset)
    registry.register_loader("tabular", load_tabular_dataset)

    return registry
