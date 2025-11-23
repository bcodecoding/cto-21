"""Lightweight dataset loaders shared by the training engine."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split


class CSVDataset(Dataset):
    """Minimal CSV backed dataset for small experiments."""

    def __init__(
        self,
        csv_path: str | Path,
        feature_columns: Optional[Sequence[str]] = None,
        label_column: Optional[str] = "label",
        label_mapping: Optional[dict[str, int]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("CSV file must include header row")
            columns = reader.fieldnames
            if feature_columns is None:
                feature_columns = [c for c in columns if c != label_column]

            self._feature_columns = list(feature_columns)
            self._label_column = label_column
            self._dtype = dtype
            self._samples: List[tuple[torch.Tensor, torch.Tensor] | torch.Tensor] = []
            mapping = dict(label_mapping) if label_mapping is not None else {}

            for row in reader:
                features = torch.tensor(
                    [float(row[col]) for col in self._feature_columns], dtype=dtype
                )
                if label_column is None:
                    self._samples.append(features)
                    continue

                label_value = row[label_column]
                if label_value not in mapping:
                    mapping[label_value] = len(mapping)
                encoded = mapping[label_value]
                label_tensor = torch.tensor(encoded, dtype=torch.long)
                self._samples.append((features, label_tensor))

        self.label_mapping = mapping if mapping else None

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._samples)

    def __getitem__(self, index: int):  # pragma: no cover - trivial
        return self._samples[index]

    @property
    def feature_columns(self) -> Sequence[str]:
        return self._feature_columns


def create_data_loaders(
    dataset: Dataset,
    batch_size: int,
    val_ratio: float = 0.2,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 13,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Split a dataset into train/val loaders."""

    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be within [0, 1)")

    if val_ratio == 0:
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        return train_loader, None

    total_len = len(dataset)
    val_len = max(1, int(total_len * val_ratio))
    train_len = total_len - val_len
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader
