"""Configuration objects for dataset declarations via JSON/YAML."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import json


@dataclass
class DatasetConfig:
    """Base configuration for datasets."""

    name: str
    dataset_type: str
    description: str = ""
    path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert config to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> DatasetConfig:
        """Create config from dictionary."""
        config_type = data.get("dataset_type", "")
        if config_type == "image":
            return ImageDatasetConfig.from_dict(data)
        elif config_type == "text":
            return TextDatasetConfig.from_dict(data)
        elif config_type == "tabular":
            return TabularDatasetConfig.from_dict(data)
        else:
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, json_str: str) -> DatasetConfig:
        """Create config from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_file(cls, path: str | Path) -> DatasetConfig:
        """Create config from JSON file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


@dataclass
class ImageDatasetConfig(DatasetConfig):
    """Configuration for image datasets."""

    dataset_type: str = "image"
    image_size: tuple[int, int] = (224, 224)
    normalize: bool = True
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    class_mapping: Optional[Dict[str, int]] = None

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        d = asdict(self)
        # Convert tuples to lists for JSON serialization
        if "image_size" in d:
            d["image_size"] = list(d["image_size"])
        if "mean" in d:
            d["mean"] = list(d["mean"])
        if "std" in d:
            d["std"] = list(d["std"])
        return d

    @classmethod
    def from_dict(cls, data: dict) -> ImageDatasetConfig:
        """Create config from dictionary."""
        data = dict(data)  # Make a copy to avoid modifying original
        # Convert lists back to tuples if needed
        if "image_size" in data and isinstance(data["image_size"], list):
            data["image_size"] = tuple(data["image_size"])
        if "mean" in data and isinstance(data["mean"], list):
            data["mean"] = tuple(data["mean"])
        if "std" in data and isinstance(data["std"], list):
            data["std"] = tuple(data["std"])
        # Filter out unknown fields
        valid_fields = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in data.items() if k in valid_fields})


@dataclass
class TextDatasetConfig(DatasetConfig):
    """Configuration for text datasets."""

    dataset_type: str = "text"
    format: str = "jsonl"  # 'jsonl' or 'csv'
    prompt_column: str = "prompt"
    label_column: str = "label"
    max_length: Optional[int] = None
    tokenizer: str = "simple"  # 'simple' or other tokenizer names
    vocab_size: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict) -> TextDatasetConfig:
        """Create config from dictionary."""
        data = dict(data)
        valid_fields = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in data.items() if k in valid_fields})


@dataclass
class TabularDatasetConfig(DatasetConfig):
    """Configuration for tabular datasets."""

    dataset_type: str = "tabular"
    feature_columns: Optional[List[str]] = None
    label_column: Optional[str] = None
    label_mapping: Optional[Dict[str, int]] = None
    normalize: bool = True
    dtype: str = "float32"

    @classmethod
    def from_dict(cls, data: dict) -> TabularDatasetConfig:
        """Create config from dictionary."""
        data = dict(data)
        valid_fields = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in data.items() if k in valid_fields})
