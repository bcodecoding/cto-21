"""Text dataset loader for JSON/CSV text datasets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import torch

from data_pipeline.base import BaseDataset, Compose
from data_pipeline.configs import TextDatasetConfig
from data_pipeline.transforms import SimpleTokenizer


class TextDataset(BaseDataset):
    """PyTorch Dataset for loading text data from JSON or CSV files.

    Expected format:
    - JSONL: {"prompt": "...", "label": "..."}
    - CSV: prompt,label columns
    """

    def __init__(
        self,
        file_path: str | Path,
        name: str = "text_dataset",
        description: str = "",
        format: str = "jsonl",
        prompt_column: str = "prompt",
        label_column: str = "label",
        max_length: Optional[int] = None,
        tokenizer: Optional[SimpleTokenizer] = None,
        label_mapping: Optional[dict[str, int]] = None,
    ) -> None:
        """Initialize text dataset.

        Args:
            file_path: Path to JSONL or CSV file
            name: Dataset name
            description: Dataset description
            format: 'jsonl' or 'csv'
            prompt_column: Column name for text/prompt
            label_column: Column name for labels
            max_length: Maximum sequence length
            tokenizer: Optional SimpleTokenizer instance
            label_mapping: Optional custom label to index mapping
        """
        super().__init__(name, description)
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        self.format = format.lower()
        self.prompt_column = prompt_column
        self.label_column = label_column
        self.max_length = max_length
        self.label_mapping = label_mapping or {}
        self._samples: list[tuple[str, int]] = []

        # Load data from file
        self._load_data()

        # Build tokenizer if not provided
        if tokenizer is None:
            tokenizer = SimpleTokenizer(max_length=max_length)
            # Build vocab from prompts
            prompts = [prompt for prompt, _ in self._samples]
            tokenizer.build_vocab(prompts)

        self.tokenizer = tokenizer

    def _load_data(self) -> None:
        """Load data from file (JSONL or CSV)."""
        if self.format == "jsonl":
            self._load_jsonl()
        elif self.format == "csv":
            self._load_csv()
        else:
            raise ValueError(f"Unsupported format: {self.format}")

    def _load_jsonl(self) -> None:
        """Load data from JSONL file."""
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                prompt = record.get(self.prompt_column, "")
                label_str = record.get(self.label_column, "unknown")

                if label_str not in self.label_mapping:
                    self.label_mapping[label_str] = len(self.label_mapping)

                label_idx = self.label_mapping[label_str]
                self._samples.append((prompt, label_idx))

    def _load_csv(self) -> None:
        """Load data from CSV file."""
        with open(self.file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = row.get(self.prompt_column, "")
                label_str = row.get(self.label_column, "unknown")

                if label_str not in self.label_mapping:
                    self.label_mapping[label_str] = len(self.label_mapping)

                label_idx = self.label_mapping[label_str]
                self._samples.append((prompt, label_idx))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Get tokenized text and label.

        Returns:
            Tuple of (token_ids_tensor, label_index)
        """
        prompt, label = self._samples[index]
        token_ids = self.tokenizer(prompt)
        return token_ids, label

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

    @classmethod
    def from_config(cls, config: TextDatasetConfig) -> TextDataset:
        """Create dataset from configuration.

        Args:
            config: TextDatasetConfig instance

        Returns:
            TextDataset instance
        """
        return cls(
            file_path=config.path,
            name=config.name,
            description=config.description,
            format=config.format,
            prompt_column=config.prompt_column,
            label_column=config.label_column,
            max_length=config.max_length,
            label_mapping=config.label_mapping,
        )
