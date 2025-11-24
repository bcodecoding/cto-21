"""
Demonstration script for the data pipeline module.

This script shows how to use the modular data loading system with:
- Image datasets (folder-based with class labels)
- Text datasets (JSONL/CSV with prompt and label)
- Tabular datasets (CSV with numeric features)
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_pipeline import (
    DatasetConfig,
    DatasetRegistry,
    ImageDatasetConfig,
    TabularDatasetConfig,
    TextDatasetConfig,
)
from data_pipeline.registries import create_default_registry


def demo_image_dataset():
    """Demo: Load and iterate through image dataset."""
    print("\n" + "=" * 60)
    print("DEMO 1: Image Dataset")
    print("=" * 60)

    # Create config
    config = ImageDatasetConfig(
        name="sample_animals",
        path="examples/data/images",
        description="Sample images of cats and dogs",
        image_size=(224, 224),
        normalize=True,
    )

    print(f"Loading image dataset from: {config.path}")
    print(f"Image size: {config.image_size}")
    print(f"Normalize: {config.normalize}")

    # Load dataset via registry
    registry = create_default_registry()
    dataset = registry.load_from_config(config)

    print(f"Dataset loaded: {dataset}")
    print(f"Number of samples: {len(dataset)}")
    print(f"Classes: {dataset.get_class_names()}")

    # Create dataloader
    loader = registry.create_dataloader(dataset, batch_size=2, shuffle=False)

    # Iterate through batches
    print("\nIterating through batches:")
    for batch_idx, (images, labels) in enumerate(loader):
        print(f"  Batch {batch_idx}:")
        print(f"    Images shape: {images.shape}")
        print(f"    Labels: {labels}")
        print(f"    Label names: {[dataset.get_class_name(l.item()) for l in labels]}")
        if batch_idx >= 1:  # Show first 2 batches
            break

    print("\n✓ Image dataset demo completed successfully!")


def demo_text_dataset():
    """Demo: Load and iterate through text dataset."""
    print("\n" + "=" * 60)
    print("DEMO 2: Text Dataset")
    print("=" * 60)

    # Create config
    config = TextDatasetConfig(
        name="sample_sentiments",
        path="examples/data/text_samples.jsonl",
        description="Sample text with sentiment labels",
        format="jsonl",
        prompt_column="prompt",
        label_column="label",
        max_length=50,
    )

    print(f"Loading text dataset from: {config.path}")
    print(f"Format: {config.format}")
    print(f"Max sequence length: {config.max_length}")

    # Load dataset via registry
    registry = create_default_registry()
    dataset = registry.load_from_config(config)

    print(f"Dataset loaded: {dataset}")
    print(f"Number of samples: {len(dataset)}")
    print(f"Classes: {dataset.get_label_names()}")
    print(f"Vocabulary size: {len(dataset.tokenizer.vocab)}")

    # Create dataloader
    loader = registry.create_dataloader(dataset, batch_size=3, shuffle=False)

    # Iterate through batches
    print("\nIterating through batches:")
    for batch_idx, (token_ids, labels) in enumerate(loader):
        print(f"  Batch {batch_idx}:")
        print(f"    Token IDs shape: {token_ids.shape}")
        print(f"    Labels: {labels}")
        print(f"    Label names: {[dataset.get_label_name(l.item()) for l in labels]}")
        if batch_idx >= 0:  # Show first batch
            break

    print("\n✓ Text dataset demo completed successfully!")


def demo_tabular_dataset():
    """Demo: Load and iterate through tabular dataset."""
    print("\n" + "=" * 60)
    print("DEMO 3: Tabular Dataset")
    print("=" * 60)

    # Create config
    config = TabularDatasetConfig(
        name="sample_loans",
        path="examples/data/tabular_sample.csv",
        description="Sample tabular data for loan approval",
        feature_columns=["age", "income", "credit_score", "employment_years"],
        label_column="label",
        normalize=True,
    )

    print(f"Loading tabular dataset from: {config.path}")
    print(f"Features: {config.feature_columns}")
    print(f"Label column: {config.label_column}")
    print(f"Normalize: {config.normalize}")

    # Load dataset via registry
    registry = create_default_registry()
    dataset = registry.load_from_config(config)

    print(f"Dataset loaded: {dataset}")
    print(f"Number of samples: {len(dataset)}")
    print(f"Number of features: {len(dataset.feature_columns)}")
    print(f"Classes: {dataset.get_label_names()}")

    # Create dataloader
    loader = registry.create_dataloader(dataset, batch_size=3, shuffle=False)

    # Iterate through batches
    print("\nIterating through batches:")
    for batch_idx, (features, labels) in enumerate(loader):
        print(f"  Batch {batch_idx}:")
        print(f"    Features shape: {features.shape}")
        print(f"    Feature range: [{features.min():.3f}, {features.max():.3f}]")
        print(f"    Labels: {labels}")
        print(f"    Label names: {[dataset.get_label_name(l.item()) for l in labels]}")
        if batch_idx >= 0:  # Show first batch
            break

    print("\n✓ Tabular dataset demo completed successfully!")


def demo_config_files():
    """Demo: Load datasets from config files."""
    print("\n" + "=" * 60)
    print("DEMO 4: Loading from Config Files")
    print("=" * 60)

    config_dir = Path("examples/data/configs")
    registry = create_default_registry()

    for config_file in sorted(config_dir.glob("*.json")):
        print(f"\nLoading config from: {config_file.name}")
        config = DatasetConfig.from_file(config_file)
        print(f"  Name: {config.name}")
        print(f"  Type: {config.dataset_type}")
        print(f"  Description: {config.description}")

        # Load dataset
        dataset = registry.load_from_config(config)
        print(f"  Loaded: {dataset}")
        print(f"  Samples: {len(dataset)}")

    print("\n✓ Config file demo completed successfully!")


def demo_custom_dataloader():
    """Demo: Create custom data loaders with different batch sizes."""
    print("\n" + "=" * 60)
    print("DEMO 5: Custom DataLoaders")
    print("=" * 60)

    registry = create_default_registry()

    # Tabular dataset with different batch sizes
    config = TabularDatasetConfig(
        name="sample_loans",
        path="examples/data/tabular_sample.csv",
        label_column="label",
    )
    dataset = registry.load_from_config(config)

    for batch_size in [1, 2, 4]:
        loader = registry.create_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        num_batches = len(loader)
        print(f"  Batch size {batch_size}: {num_batches} batches")

    print("\n✓ Custom dataloader demo completed successfully!")


def main():
    """Run all demonstration scripts."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Data Pipeline Module - Demonstration".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")

    try:
        demo_image_dataset()
        demo_text_dataset()
        demo_tabular_dataset()
        demo_config_files()
        demo_custom_dataloader()

        print("\n" + "=" * 60)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nKey Features Demonstrated:")
        print("  ✓ Image dataset loading with transforms (resize, normalize)")
        print("  ✓ Text dataset loading with tokenization")
        print("  ✓ Tabular dataset loading with numeric scaling")
        print("  ✓ Configuration-based dataset instantiation")
        print("  ✓ PyTorch DataLoader integration")
        print("  ✓ Batch iteration for all dataset types")
        print()

    except Exception as e:
        print(f"\n✗ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
