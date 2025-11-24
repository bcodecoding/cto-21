# Data Pipeline Examples

This directory contains example datasets and scripts demonstrating the data pipeline module.

## Files

- `data_pipeline_demo.py` - Complete demonstration script showing all dataset types
- `data/` - Sample datasets in multiple formats

## Sample Datasets

### Images (`data/images/`)

```
data/images/
  cat/
    cat_0.ppm
    cat_1.ppm
    cat_2.ppm
  dog/
    dog_0.ppm
    dog_1.ppm
    dog_2.ppm
```

**Format**: PPM image files (100x100 pixels, RGB)
**Classes**: 2 (cat, dog)
**Total images**: 6

**Configuration**: `data/configs/image_config.json`

### Text (`data/text_samples.jsonl`)

```jsonl
{"prompt": "This movie was absolutely fantastic", "label": "positive"}
{"prompt": "I did not like this product at all", "label": "negative"}
{"prompt": "The experience was okay nothing special", "label": "neutral"}
```

**Format**: JSONL (one JSON object per line)
**Columns**: `prompt` (text), `label` (class)
**Classes**: 3 (positive, negative, neutral)
**Total samples**: 9

**Configuration**: `data/configs/text_config.json`

### Tabular (`data/tabular_sample.csv`)

```csv
age,income,credit_score,employment_years,label
25,45000,650,2,approved
35,85000,750,8,approved
22,32000,580,1,denied
```

**Format**: CSV with headers
**Columns**: `age`, `income`, `credit_score`, `employment_years` (features), `label` (target)
**Classes**: 2 (approved, denied)
**Total samples**: 12

**Configuration**: `data/configs/tabular_config.json`

## Running the Demo

```bash
# From project root
python examples/data_pipeline_demo.py
```

Expected output:
```
============================================================
DEMO 1: Image Dataset
============================================================
Loading image dataset from: examples/data/images
Image size: (224, 224)
Normalize: True
Dataset loaded: ImageDataset(name=sample_animals, len=6)
Number of samples: 6
Classes: ['cat', 'dog']

Iterating through batches:
  Batch 0:
    Images shape: torch.Size([2, 3, 224, 224])
    Labels: tensor([0, 0])
    Label names: ['cat', 'cat']
  Batch 1:
    Images shape: torch.Size([2, 3, 224, 224])
    Labels: tensor([0, 1])
    Label names: ['cat', 'dog']

✓ Image dataset demo completed successfully!

...
```

## Creating Your Own Datasets

### Image Dataset

1. Create a folder structure:
```
my_images/
  class_a/
    image1.jpg
    image2.jpg
  class_b/
    image3.jpg
```

2. Load with Python:
```python
from data_pipeline import ImageDataset

dataset = ImageDataset(
    root_dir="my_images",
    image_size=(224, 224),
    normalize=True,
)
```

3. Or with config:
```python
from data_pipeline import ImageDatasetConfig, create_default_registry

config = ImageDatasetConfig(
    name="my_images",
    path="my_images",
    image_size=(224, 224),
)

registry = create_default_registry()
dataset = registry.load_from_config(config)
```

### Text Dataset

1. Create a JSONL file:
```jsonl
{"prompt": "sample text 1", "label": "class_a"}
{"prompt": "sample text 2", "label": "class_b"}
```

Or CSV:
```csv
prompt,label
"sample text 1",class_a
"sample text 2",class_b
```

2. Load with Python:
```python
from data_pipeline import TextDataset

dataset = TextDataset(
    file_path="my_text.jsonl",
    format="jsonl",
    max_length=128,
)
```

### Tabular Dataset

1. Create a CSV file:
```csv
feature1,feature2,feature3,label
0.5,0.6,0.7,target_a
0.1,0.2,0.3,target_b
```

2. Load with Python:
```python
from data_pipeline import TabularDataset

dataset = TabularDataset(
    csv_path="my_data.csv",
    feature_columns=["feature1", "feature2", "feature3"],
    label_column="label",
    normalize=True,
)
```

## Configuration Files

All datasets can be defined via JSON configuration files.

### Image Config Example

```json
{
  "name": "my_images",
  "dataset_type": "image",
  "description": "My image classification dataset",
  "path": "path/to/images",
  "image_size": [224, 224],
  "normalize": true,
  "mean": [0.485, 0.456, 0.406],
  "std": [0.229, 0.224, 0.225],
  "metadata": {
    "num_classes": 2,
    "classes": ["class_a", "class_b"]
  }
}
```

### Text Config Example

```json
{
  "name": "my_text",
  "dataset_type": "text",
  "description": "My text classification dataset",
  "path": "path/to/data.jsonl",
  "format": "jsonl",
  "prompt_column": "text",
  "label_column": "label",
  "max_length": 128,
  "tokenizer": "simple",
  "metadata": {
    "num_classes": 3,
    "classes": ["class_a", "class_b", "class_c"]
  }
}
```

### Tabular Config Example

```json
{
  "name": "my_data",
  "dataset_type": "tabular",
  "description": "My tabular dataset",
  "path": "path/to/data.csv",
  "feature_columns": ["age", "income", "score"],
  "label_column": "target",
  "normalize": true,
  "dtype": "float32",
  "metadata": {
    "num_classes": 2,
    "classes": ["approved", "denied"]
  }
}
```

## Loading from Configs

```python
from data_pipeline import DatasetConfig, create_default_registry

# Load configuration
config = DatasetConfig.from_file("configs/my_dataset.json")

# Create registry
registry = create_default_registry()

# Load dataset
dataset = registry.load_from_config(config)

# Create DataLoader
loader = registry.create_dataloader(dataset, batch_size=32)

# Use in training
for features, labels in loader:
    # Your training code here
    pass
```

## Using with Training Loop

### Complete Example

```python
from data_pipeline import TabularDatasetConfig, create_default_registry
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

# Load dataset
config = TabularDatasetConfig(
    name="loans",
    path="examples/data/tabular_sample.csv",
    feature_columns=["age", "income", "credit_score", "employment_years"],
    label_column="label",
    normalize=True,
)

registry = create_default_registry()
dataset = registry.load_from_config(config)
loader = registry.create_dataloader(dataset, batch_size=4, shuffle=True)

# Define model
model = nn.Linear(4, 2)  # 4 input features, 2 output classes
optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

# Training
for epoch in range(10):
    for features, labels in loader:
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(features)
        loss = loss_fn(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch + 1}: loss={loss.item():.4f}")
```

## Extending the Data Pipeline

### Custom Dataset Type

```python
from data_pipeline.base import BaseDataset, DatasetLoader
from data_pipeline import DatasetRegistry

class MyCustomDataset(BaseDataset):
    def __init__(self, data_path, name="custom"):
        super().__init__(name)
        # Load your custom data
        self._samples = []  # Your samples
    
    def __len__(self):
        return len(self._samples)
    
    def __getitem__(self, index):
        return self._samples[index]

# Register with the system
registry = DatasetRegistry()

def load_custom(config):
    return MyCustomDataset(config.path)

registry.register_loader("custom", load_custom)

# Use like any other dataset
# config = CustomDatasetConfig(...)
# dataset = registry.load_from_config(config)
```

## Common Issues and Solutions

### "No images found" Error

Ensure your directory structure is correct:
```
root_dir/
  class1/
    image.jpg
  class2/
    image.jpg
```

Not:
```
root_dir/
  image1.jpg
  image2.jpg
```

### Text Tokenization Issues

The simple tokenizer splits on spaces. For better tokenization, implement a custom tokenizer:

```python
from data_pipeline.transforms import Transform
import torch

class BetterTokenizer(Transform):
    def __init__(self, max_length=128):
        self.max_length = max_length
    
    def __call__(self, text):
        # Your tokenization logic
        tokens = []  # Your tokens
        return torch.tensor(tokens, dtype=torch.long)
```

### Memory Issues with Large Datasets

Use DataLoader's `num_workers` for parallel data loading:

```python
loader = registry.create_dataloader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,  # Use 4 worker processes
)
```

## Troubleshooting

### Import Errors

Ensure the data_pipeline module is in your Python path:

```bash
# From project root
python -c "from data_pipeline import DatasetRegistry; print('OK')"
```

### File Not Found Errors

Use absolute paths or ensure working directory:

```python
from pathlib import Path

# Use absolute paths
dataset = ImageDataset(
    root_dir=Path(__file__).parent / "data" / "images"
)
```

### Data Type Mismatches

Ensure feature columns are numeric for tabular data:

```python
# Correct
config = TabularDatasetConfig(
    feature_columns=["age", "income", "score"],  # Numeric
    label_column="outcome",  # Can be strings
)

# Incorrect - only select numeric columns
config = TabularDatasetConfig(
    feature_columns=["age", "name", "income"],  # 'name' is a string!
    label_column="outcome",
)
```

## Performance Tips

1. **Use num_workers** for data loading parallelization
2. **Pre-process and cache** large datasets when possible
3. **Use appropriate batch sizes** (32-128 typical)
4. **Normalize/scale data** for better training stability
5. **Use shuffle=True** for training but not validation

## See Also

- `../../data_pipeline/README.md` - Complete API documentation
- `../../tests/test_data_pipeline.py` - Comprehensive test examples
