# Data Pipeline Module

A modular, extensible data loading and preprocessing layer for machine learning pipelines. Provides abstractions for dataset management, loaders for multiple data modalities (images, text, tabular), and PyTorch integration.

## Features

- **Modular Dataset Loaders**: Specialized loaders for different data types
  - `ImageDataset`: Folder-based image classification datasets
  - `TextDataset`: Text data from JSONL/CSV with tokenization
  - `TabularDataset`: CSV-based tabular data with numeric scaling

- **Configuration-Based Instantiation**: Define datasets via JSON/YAML configs

- **Built-in Transforms**: Pre-processing operations
  - Image: resize, normalize
  - Text: tokenization
  - Tabular: numeric scaling (min-max, standardization)

- **Registry System**: Central management of datasets and loaders

- **PyTorch Integration**: Full `Dataset`/`DataLoader` compatibility

## Quick Start

### Load an Image Dataset

```python
from data_pipeline import ImageDatasetConfig, create_default_registry

# Define configuration
config = ImageDatasetConfig(
    name="my_images",
    path="path/to/images",  # Folder with class subdirectories
    image_size=(224, 224),
    normalize=True,
)

# Create registry and load dataset
registry = create_default_registry()
dataset = registry.load_from_config(config)

# Create DataLoader
loader = registry.create_dataloader(dataset, batch_size=32)

# Iterate batches
for images, labels in loader:
    # images: (B, 3, 224, 224)
    # labels: (B,)
    pass
```

### Load a Text Dataset

```python
from data_pipeline import TextDatasetConfig, create_default_registry

# Define configuration
config = TextDatasetConfig(
    name="my_text",
    path="path/to/data.jsonl",
    format="jsonl",
    prompt_column="text",
    label_column="label",
    max_length=128,
)

# Load dataset
registry = create_default_registry()
dataset = registry.load_from_config(config)
loader = registry.create_dataloader(dataset, batch_size=16)

# Iterate
for token_ids, labels in loader:
    # token_ids: (B, max_length)
    # labels: (B,)
    pass
```

### Load a Tabular Dataset

```python
from data_pipeline import TabularDatasetConfig, create_default_registry

# Define configuration
config = TabularDatasetConfig(
    name="my_data",
    path="path/to/data.csv",
    feature_columns=["age", "income", "score"],
    label_column="target",
    normalize=True,
)

# Load dataset
registry = create_default_registry()
dataset = registry.load_from_config(config)
loader = registry.create_dataloader(dataset, batch_size=32)

# Iterate
for features, labels in loader:
    # features: (B, num_features) - normalized to [0, 1]
    # labels: (B,)
    pass
```

### Load from Config File

```python
from data_pipeline import DatasetConfig, create_default_registry

# Load configuration from JSON
config = DatasetConfig.from_file("path/to/config.json")

# Load dataset
registry = create_default_registry()
dataset = registry.load_from_config(config)
```

## Dataset Formats

### Image Dataset

**Directory Structure**:
```
root_dir/
  class1/
    image1.jpg
    image2.png
    ...
  class2/
    image3.jpg
    ...
```

**Supported Formats**: `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.ppm`

**Configuration**:
```json
{
  "name": "my_images",
  "dataset_type": "image",
  "path": "path/to/images",
  "image_size": [224, 224],
  "normalize": true,
  "mean": [0.485, 0.456, 0.406],
  "std": [0.229, 0.224, 0.225]
}
```

### Text Dataset

**JSONL Format** (one JSON object per line):
```jsonl
{"prompt": "This is great", "label": "positive"}
{"prompt": "Not good", "label": "negative"}
```

**CSV Format**:
```csv
prompt,label
"This is great",positive
"Not good",negative
```

**Configuration**:
```json
{
  "name": "my_text",
  "dataset_type": "text",
  "path": "path/to/data.jsonl",
  "format": "jsonl",
  "prompt_column": "prompt",
  "label_column": "label",
  "max_length": 128,
  "tokenizer": "simple"
}
```

### Tabular Dataset

**CSV Format** (standard with headers):
```csv
age,income,score,target
25,45000,650,approved
35,85000,750,approved
```

**Configuration**:
```json
{
  "name": "my_data",
  "dataset_type": "tabular",
  "path": "path/to/data.csv",
  "feature_columns": ["age", "income", "score"],
  "label_column": "target",
  "normalize": true,
  "dtype": "float32"
}
```

## Usage Examples

### Example 1: Image Classification

```python
from data_pipeline import ImageDataset
from torch.utils.data import DataLoader

# Create dataset directly
dataset = ImageDataset(
    root_dir="path/to/images",
    image_size=(224, 224),
    normalize=True,
)

# Get info
print(f"Classes: {dataset.get_class_names()}")
print(f"Number of images: {len(dataset)}")

# Create dataloader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for epoch in range(10):
    for images, labels in loader:
        # Your training code here
        pass
```

### Example 2: Text Classification

```python
from data_pipeline import TextDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = TextDataset(
    file_path="path/to/data.jsonl",
    format="jsonl",
    max_length=128,
)

# Get info
print(f"Labels: {dataset.get_label_names()}")
print(f"Vocabulary size: {len(dataset.tokenizer.vocab)}")

# Create dataloader
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Training loop
for tokens, labels in loader:
    # Your training code here
    pass
```

### Example 3: Tabular/Regression

```python
from data_pipeline import TabularDataset
from torch.utils.data import DataLoader

# Create dataset with normalization
dataset = TabularDataset(
    csv_path="path/to/data.csv",
    feature_columns=["age", "income", "score"],
    label_column="target",
    normalize=True,  # Applies min-max scaling
)

# Get info
print(f"Features: {dataset.feature_columns}")
print(f"Number of samples: {len(dataset)}")

# Create dataloader
loader = DataLoader(dataset, batch_size=32)

# Training loop
for features, labels in loader:
    # Your training code here
    pass
```

### Example 4: Registry Management

```python
from data_pipeline import DatasetRegistry, ImageDatasetConfig

# Create registry
registry = DatasetRegistry()

# Register custom loaders
def my_custom_loader(config):
    # Your custom loading logic
    pass

registry.register_loader("custom_type", my_custom_loader)

# Register datasets
config = ImageDatasetConfig(name="my_images", path="...")
dataset = ImageDataset.from_config(config)
registry.register_dataset("my_images", dataset, config)

# Retrieve and use
dataset = registry.get_dataset("my_images")
loader = registry.create_dataloader(dataset, batch_size=32)
```

### Example 5: Custom Transforms

```python
from data_pipeline import ImageDataset, Compose, ImageResize, ImageNormalize

# Create custom transforms pipeline
transforms = Compose([
    ImageResize((256, 256)),
    ImageNormalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
])

# Pass to dataset
dataset = ImageDataset(
    root_dir="path/to/images",
    transforms=transforms,
)
```

## Configuration Objects

### ImageDatasetConfig

```python
from data_pipeline import ImageDatasetConfig

config = ImageDatasetConfig(
    name: str,                          # Dataset name
    path: str,                          # Root directory path
    description: str = "",              # Description
    image_size: tuple[int, int] = (224, 224),  # Target size
    normalize: bool = True,             # Apply normalization
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
    class_mapping: Optional[Dict] = None,
)
```

### TextDatasetConfig

```python
from data_pipeline import TextDatasetConfig

config = TextDatasetConfig(
    name: str,                          # Dataset name
    path: str,                          # File path
    description: str = "",              # Description
    format: str = "jsonl",              # 'jsonl' or 'csv'
    prompt_column: str = "prompt",      # Text column name
    label_column: str = "label",        # Label column name
    max_length: Optional[int] = None,   # Max sequence length
    tokenizer: str = "simple",          # Tokenizer type
    vocab_size: Optional[int] = None,
)
```

### TabularDatasetConfig

```python
from data_pipeline import TabularDatasetConfig

config = TabularDatasetConfig(
    name: str,                                  # Dataset name
    path: str,                                  # CSV file path
    description: str = "",                      # Description
    feature_columns: Optional[List[str]] = None,
    label_column: Optional[str] = None,
    normalize: bool = True,                     # Apply scaling
    dtype: str = "float32",                     # Data type
    label_mapping: Optional[Dict] = None,
)
```

## Transforms

### Image Transforms

```python
from data_pipeline.transforms import (
    ImageResize,      # Resize to (H, W)
    ImageNormalize,   # Normalize with mean/std
)

# Usage
resize = ImageResize((224, 224))
normalize = ImageNormalize(mean=(0.485, 0.456, 0.406))

image = resize(image)
image = normalize(image)
```

### Text Transforms

```python
from data_pipeline.transforms import SimpleTokenizer

# Build tokenizer from texts
tokenizer = SimpleTokenizer(max_length=128)
tokenizer.build_vocab(["hello world", "goodbye world"])

# Tokenize text
tokens = tokenizer("hello world")  # Returns torch.Tensor
```

### Numeric Transforms

```python
from data_pipeline.transforms import NumericScaler

# Min-max scaling
scaler = NumericScaler(method="minmax")
scaler.fit(training_data)
scaled = scaler(sample)

# Standardization
scaler = NumericScaler(method="standard")
scaler.fit(training_data)
scaled = scaler(sample)
```

## Module Structure

```
data_pipeline/
  __init__.py              # Package exports
  base.py                  # Base classes (BaseDataset, Transform)
  configs.py               # Configuration dataclasses
  transforms.py            # Transform implementations
  registries.py            # Dataset registry
  loaders/
    __init__.py
    image_loader.py        # ImageDataset
    text_loader.py         # TextDataset
    tabular_loader.py      # TabularDataset
```

## API Reference

### ImageDataset

```python
class ImageDataset(BaseDataset):
    def __init__(
        self,
        root_dir: str | Path,
        name: str = "image_dataset",
        image_size: tuple[int, int] = (224, 224),
        normalize: bool = True,
        ...
    ) -> None:
        """Load images from folder structure."""
    
    def get_class_names(self) -> list[str]:
        """Get all class names."""
    
    @classmethod
    def from_config(cls, config: ImageDatasetConfig) -> ImageDataset:
        """Create from configuration."""
```

### TextDataset

```python
class TextDataset(BaseDataset):
    def __init__(
        self,
        file_path: str | Path,
        format: str = "jsonl",
        prompt_column: str = "prompt",
        label_column: str = "label",
        max_length: Optional[int] = None,
        ...
    ) -> None:
        """Load text from JSONL/CSV file."""
    
    def get_label_names(self) -> list[str]:
        """Get all label names."""
    
    @classmethod
    def from_config(cls, config: TextDatasetConfig) -> TextDataset:
        """Create from configuration."""
```

### TabularDataset

```python
class TabularDataset(BaseDataset):
    def __init__(
        self,
        csv_path: str | Path,
        feature_columns: Optional[Sequence[str]] = None,
        label_column: Optional[str] = None,
        normalize: bool = True,
        ...
    ) -> None:
        """Load tabular data from CSV file."""
    
    def get_label_names(self) -> list[str]:
        """Get all label names."""
    
    @classmethod
    def from_config(cls, config: TabularDatasetConfig) -> TabularDataset:
        """Create from configuration."""
```

### DatasetRegistry

```python
class DatasetRegistry:
    def register_dataset(
        self,
        name: str,
        dataset: BaseDataset,
        config: Optional[DatasetConfig] = None,
    ) -> None:
        """Register a dataset."""
    
    def get_dataset(self, name: str) -> Optional[BaseDataset]:
        """Retrieve registered dataset."""
    
    def load_from_config(self, config: DatasetConfig) -> BaseDataset:
        """Load dataset from config."""
    
    def create_dataloader(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        """Create PyTorch DataLoader."""
    
    def list_datasets(self) -> list[str]:
        """List all registered datasets."""
```

## Error Handling

The module includes comprehensive error checking:

```python
from data_pipeline import ImageDataset

# File not found
try:
    dataset = ImageDataset("/nonexistent/path")
except FileNotFoundError as e:
    print(f"Path error: {e}")

# No images found
try:
    dataset = ImageDataset("/empty/directory")
except ValueError as e:
    print(f"Data error: {e}")

# Invalid image file
try:
    image, label = dataset[0]
except IOError as e:
    print(f"Load error: {e}")
```

## Performance Notes

- **Image loading**: Resizing and normalization happen on-the-fly during iteration
- **Text tokenization**: Vocabulary is built once during dataset initialization
- **Numeric scaling**: Statistics computed once during dataset initialization
- **Batch processing**: All datasets work with PyTorch DataLoader's `num_workers` for parallel loading

## Extensions

To add custom dataset types:

```python
from data_pipeline.base import BaseDataset
from data_pipeline import DatasetRegistry

class MyCustomDataset(BaseDataset):
    def __init__(self, ...):
        super().__init__(name, description)
        # Your initialization
    
    def __len__(self):
        return len(self._samples)
    
    def __getitem__(self, index):
        return self._samples[index]

# Register with registry
def load_custom(config):
    return MyCustomDataset(...)

registry = DatasetRegistry()
registry.register_loader("custom", load_custom)
```

## See Also

- `examples/data_pipeline_demo.py` - Complete usage examples
- `tests/test_data_pipeline.py` - Unit and integration tests
- `examples/data/configs/` - Sample configuration files
