"""Tabular dataset loader for CSV datasets."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import torch

from data_pipeline.base import BaseDataset
from data_pipeline.configs import TabularDatasetConfig
from data_pipeline.transforms import NumericScaler


class TabularDataset(BaseDataset):
    """PyTorch Dataset for loading tabular data from CSV files.

    Expects CSV with header row and numeric feature columns.
    """

    def __init__(
        self,
        csv_path: str | Path,
        name: str = "tabular_dataset",
        description: str = "",
        feature_columns: Sequence[str] | None = None,
        label_column: str | None = None,
        label_mapping: dict[str, int] | None = None,
        normalize: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize tabular dataset.

        Args:
            csv_path: Path to CSV file
            name: Dataset name
            description: Dataset description
            feature_columns: Sequence of column names to use as features
            label_column: Column name for labels (optional)
            label_mapping: Optional custom label to index mapping
            normalize: Whether to normalize numeric features
            dtype: Torch data type for features
        """
        super().__init__(name, description)
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        self.label_column = label_column
        self.label_mapping = dict(label_mapping) if label_mapping else {}
        self.normalize = normalize
        self.dtype = dtype

        # Load data
        with self.csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV must have header row")

            columns = reader.fieldnames
            # Use all columns except label as features if not specified
            if feature_columns is None:
                feature_columns = [c for c in columns if c != label_column]

            self._feature_columns = list(feature_columns)
            self._samples: list[tuple[torch.Tensor, torch.Tensor] | torch.Tensor] = []

            for row in reader:
                try:
                    features = torch.tensor(
                        [float(row[col]) for col in self._feature_columns],
                        dtype=dtype,
                    )
                except (KeyError, ValueError) as e:
                    raise ValueError(f"Error parsing row {row}: {e}") from e

                if label_column is None:
                    self._samples.append(features)
                else:
                    label_value = row[label_column]
                    if label_value not in self.label_mapping:
                        self.label_mapping[label_value] = len(self.label_mapping)
                    label_idx = self.label_mapping[label_value]
                    label_tensor = torch.tensor(label_idx, dtype=torch.long)
                    self._samples.append((features, label_tensor))

        if not self._samples:
            raise ValueError(f"No valid samples found in {csv_path}")

        # Build scaler if normalization is enabled
        if normalize:
            # Collect all feature data
            if label_column is None:
                feature_data = torch.stack(self._samples)
            else:
                feature_data = torch.stack([f for f, _ in self._samples])

            self.scaler = NumericScaler(method="minmax")
            self.scaler.fit(feature_data)
        else:
            self.scaler = None

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int):
        """Get features and optional label.

        Returns:
            Features tensor or (features, label) tuple
        """
        sample = self._samples[index]

        if isinstance(sample, tuple):
            features, label = sample
            if self.scaler:
                features = self.scaler(features)
            return features, label
        else:
            features = sample
            if self.scaler:
                features = self.scaler(features)
            return features

    def get_label_name(self, label_idx: int) -> str:
        """Get label name from index.

        Args:
            label_idx: Label index

        Returns:
            Label name
        """
        for name, idx in self.label_mapping.items():
            if idx == label_idx:
                return name
        return "unknown"

    def get_label_names(self) -> list[str]:
        """Get all label names in order.

        Returns:
            List of label names
        """
        reverse_map = {idx: name for name, idx in self.label_mapping.items()}
        return [reverse_map[i] for i in sorted(reverse_map.keys())]

    @property
    def feature_columns(self) -> Sequence[str]:
        """Get feature column names.

        Returns:
            Sequence of column names
        """
        return self._feature_columns

    @classmethod
    def from_config(cls, config: TabularDatasetConfig) -> TabularDataset:
        """Create dataset from configuration.

        Args:
            config: TabularDatasetConfig instance

        Returns:
            TabularDataset instance
        """
        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "float16": torch.float16,
        }
        dtype = dtype_map.get(config.dtype, torch.float32)

        return cls(
            csv_path=config.path,
            name=config.name,
            description=config.description,
            feature_columns=config.feature_columns,
            label_column=config.label_column,
            label_mapping=config.label_mapping,
            normalize=config.normalize,
            dtype=dtype,
        )
