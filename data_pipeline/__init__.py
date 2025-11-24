"""Modular data loading and preprocessing layer for ML training."""

from data_pipeline.configs import (
    DatasetConfig,
    ImageDatasetConfig,
    TabularDatasetConfig,
    TextDatasetConfig,
)
from data_pipeline.loaders.image_loader import ImageDataset
from data_pipeline.loaders.tabular_loader import TabularDataset
from data_pipeline.loaders.text_loader import TextDataset
from data_pipeline.registries import DatasetRegistry

__all__ = [
    "DatasetConfig",
    "ImageDatasetConfig",
    "TextDatasetConfig",
    "TabularDatasetConfig",
    "ImageDataset",
    "TextDataset",
    "TabularDataset",
    "DatasetRegistry",
]
