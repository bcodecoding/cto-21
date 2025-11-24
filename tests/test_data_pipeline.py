"""Unit and integration tests for the data pipeline module."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_pipeline import (
    DatasetConfig,
    ImageDataset,
    ImageDatasetConfig,
    TabularDataset,
    TabularDatasetConfig,
    TextDataset,
    TextDatasetConfig,
)
from data_pipeline.registries import create_default_registry
from data_pipeline.transforms import (
    ImageNormalize,
    ImageResize,
    NumericScaler,
    SimpleTokenizer,
)


class TestImageDataset(unittest.TestCase):
    """Test ImageDataset loader."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        # Create test image directories
        (self.root / "class1").mkdir()
        (self.root / "class2").mkdir()

        # Create simple PPM images
        self._create_test_image(self.root / "class1" / "img1.ppm")
        self._create_test_image(self.root / "class1" / "img2.ppm")
        self._create_test_image(self.root / "class2" / "img3.ppm")

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def _create_test_image(self, path: Path, color: tuple = (100, 150, 200)):
        """Create a simple test PPM image."""
        with open(path, "wb") as f:
            f.write(b"P6\n")
            f.write(b"32 32\n")
            f.write(b"255\n")
            for _ in range(32 * 32):
                f.write(bytes(color))

    def test_image_dataset_initialization(self):
        """Test ImageDataset initialization."""
        dataset = ImageDataset(self.root, name="test_images")
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.name, "test_images")
        self.assertIn("class1", dataset.class_to_idx)
        self.assertIn("class2", dataset.class_to_idx)

    def test_image_dataset_getitem(self):
        """Test getting an image from dataset."""
        dataset = ImageDataset(self.root, image_size=(64, 64))
        image, label = dataset[0]
        self.assertIsInstance(image, torch.Tensor)
        self.assertIsInstance(label, int)
        self.assertEqual(image.shape[0], 3)  # 3 color channels
        self.assertEqual(image.shape[1], 64)  # height
        self.assertEqual(image.shape[2], 64)  # width

    def test_image_dataset_class_mapping(self):
        """Test class name mapping."""
        dataset = ImageDataset(self.root)
        class_names = dataset.get_class_names()
        self.assertEqual(len(class_names), 2)
        self.assertIn("class1", class_names)
        self.assertIn("class2", class_names)

    def test_image_dataset_dataloader(self):
        """Test creating a DataLoader from ImageDataset."""
        dataset = ImageDataset(self.root)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        images, labels = batch
        self.assertEqual(images.shape[0], 2)  # batch size 2
        self.assertEqual(len(labels), 2)

    def test_image_dataset_from_config(self):
        """Test creating ImageDataset from config."""
        config = ImageDatasetConfig(
            name="test",
            path=str(self.root),
            image_size=(64, 64),
        )
        dataset = ImageDataset.from_config(config)
        self.assertEqual(len(dataset), 3)


class TestTextDataset(unittest.TestCase):
    """Test TextDataset loader."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        # Create test JSONL file
        self.jsonl_file = self.root / "test_data.jsonl"
        with open(self.jsonl_file, "w") as f:
            f.write('{"prompt": "hello world", "label": "positive"}\n')
            f.write('{"prompt": "goodbye world", "label": "negative"}\n')
            f.write('{"prompt": "neutral text here", "label": "neutral"}\n')

        # Create test CSV file
        self.csv_file = self.root / "test_data.csv"
        with open(self.csv_file, "w") as f:
            f.write("prompt,label\n")
            f.write("hello world,positive\n")
            f.write("goodbye world,negative\n")

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_text_dataset_jsonl(self):
        """Test TextDataset with JSONL format."""
        dataset = TextDataset(
            self.jsonl_file,
            format="jsonl",
            max_length=20,
        )
        self.assertEqual(len(dataset), 3)
        self.assertEqual(len(dataset.label_mapping), 3)

    def test_text_dataset_csv(self):
        """Test TextDataset with CSV format."""
        dataset = TextDataset(
            self.csv_file,
            format="csv",
            max_length=20,
        )
        self.assertEqual(len(dataset), 2)
        self.assertEqual(len(dataset.label_mapping), 2)

    def test_text_dataset_getitem(self):
        """Test getting text sample from dataset."""
        dataset = TextDataset(self.jsonl_file, max_length=20)
        tokens, label = dataset[0]
        self.assertIsInstance(tokens, torch.Tensor)
        self.assertIsInstance(label, int)
        self.assertEqual(tokens.dtype, torch.long)

    def test_text_dataset_tokenization(self):
        """Test tokenizer builds vocabulary."""
        dataset = TextDataset(self.jsonl_file, max_length=20)
        self.assertGreater(len(dataset.tokenizer.vocab), 0)

    def test_text_dataset_label_mapping(self):
        """Test label name retrieval."""
        dataset = TextDataset(self.jsonl_file)
        names = dataset.get_label_names()
        self.assertEqual(len(names), 3)

    def test_text_dataset_from_config(self):
        """Test creating TextDataset from config."""
        config = TextDatasetConfig(
            name="test",
            path=str(self.jsonl_file),
            format="jsonl",
            max_length=20,
        )
        dataset = TextDataset.from_config(config)
        self.assertEqual(len(dataset), 3)

    def test_text_dataset_dataloader(self):
        """Test creating a DataLoader from TextDataset."""
        dataset = TextDataset(self.jsonl_file, max_length=20)
        loader = DataLoader(dataset, batch_size=2)
        batch = next(iter(loader))
        tokens, labels = batch
        self.assertEqual(tokens.shape[0], 2)  # batch size
        self.assertEqual(len(labels), 2)


class TestTabularDataset(unittest.TestCase):
    """Test TabularDataset loader."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        # Create test CSV file
        self.csv_file = self.root / "test_data.csv"
        with open(self.csv_file, "w") as f:
            f.write("feature1,feature2,label\n")
            f.write("0.1,0.2,A\n")
            f.write("0.3,0.4,B\n")
            f.write("0.5,0.6,A\n")
            f.write("0.7,0.8,B\n")

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_tabular_dataset_initialization(self):
        """Test TabularDataset initialization."""
        dataset = TabularDataset(
            self.csv_file,
            label_column="label",
        )
        self.assertEqual(len(dataset), 4)
        self.assertEqual(len(dataset.feature_columns), 2)

    def test_tabular_dataset_getitem(self):
        """Test getting a sample from dataset."""
        dataset = TabularDataset(self.csv_file, label_column="label")
        features, label = dataset[0]
        self.assertIsInstance(features, torch.Tensor)
        self.assertIsInstance(label, torch.Tensor)
        self.assertEqual(len(features), 2)

    def test_tabular_dataset_normalization(self):
        """Test that normalization is applied."""
        dataset = TabularDataset(self.csv_file, normalize=True, label_column="label")
        features, _ = dataset[0]
        # Normalized features should be in [0, 1] after minmax scaling
        self.assertTrue(torch.all(features >= 0))
        self.assertTrue(torch.all(features <= 1))

    def test_tabular_dataset_label_mapping(self):
        """Test label mapping."""
        dataset = TabularDataset(self.csv_file, label_column="label")
        self.assertGreater(len(dataset.label_mapping), 0)
        label_names = dataset.get_label_names()
        self.assertIn("A", label_names)
        self.assertIn("B", label_names)

    def test_tabular_dataset_without_label(self):
        """Test TabularDataset without label column."""
        dataset = TabularDataset(
            self.csv_file,
            label_column=None,
        )
        features = dataset[0]
        self.assertIsInstance(features, torch.Tensor)

    def test_tabular_dataset_from_config(self):
        """Test creating TabularDataset from config."""
        config = TabularDatasetConfig(
            name="test",
            path=str(self.csv_file),
            feature_columns=["feature1", "feature2"],
            label_column="label",
        )
        dataset = TabularDataset.from_config(config)
        self.assertEqual(len(dataset), 4)

    def test_tabular_dataset_dataloader(self):
        """Test creating a DataLoader from TabularDataset."""
        dataset = TabularDataset(self.csv_file, label_column="label")
        loader = DataLoader(dataset, batch_size=2)
        batch = next(iter(loader))
        features, labels = batch
        self.assertEqual(features.shape[0], 2)  # batch size
        self.assertEqual(len(labels), 2)


class TestTransforms(unittest.TestCase):
    """Test transform implementations."""

    def test_image_resize(self):
        """Test ImageResize transform."""
        resize = ImageResize((64, 64))
        image = torch.randn(3, 128, 128)
        resized = resize(image)
        self.assertEqual(resized.shape, (3, 64, 64))

    def test_image_normalize(self):
        """Test ImageNormalize transform."""
        normalize = ImageNormalize()
        image = torch.ones(3, 64, 64) * 255.0
        normalized = normalize(image)
        # After normalization, values should be different from original
        self.assertFalse(torch.allclose(normalized, image))

    def test_simple_tokenizer(self):
        """Test SimpleTokenizer."""
        tokenizer = SimpleTokenizer(max_length=10)
        texts = ["hello world", "goodbye world", "test text"]
        tokenizer.build_vocab(texts)

        tokens = tokenizer("hello world")
        self.assertEqual(tokens.dtype, torch.long)
        self.assertLessEqual(len(tokens), 10)

    def test_numeric_scaler_minmax(self):
        """Test NumericScaler with minmax method."""
        scaler = NumericScaler(method="minmax")
        data = torch.tensor(
            [[0.0, 100.0], [10.0, 200.0], [20.0, 300.0]], dtype=torch.float32
        )
        scaler.fit(data)

        scaled = scaler(data[0])
        self.assertTrue(torch.all(scaled >= 0))
        self.assertTrue(torch.all(scaled <= 1))

    def test_numeric_scaler_standard(self):
        """Test NumericScaler with standard method."""
        scaler = NumericScaler(method="standard")
        data = torch.randn(100, 5)
        scaler.fit(data)

        scaled = scaler(data[0])
        self.assertEqual(scaled.shape, data[0].shape)


class TestDatasetConfig(unittest.TestCase):
    """Test configuration objects."""

    def test_image_config_to_dict(self):
        """Test ImageDatasetConfig to_dict."""
        config = ImageDatasetConfig(
            name="test",
            path="/path/to/images",
            image_size=(224, 224),
        )
        config_dict = config.to_dict()
        self.assertEqual(config_dict["name"], "test")
        self.assertEqual(config_dict["dataset_type"], "image")

    def test_image_config_to_json(self):
        """Test ImageDatasetConfig to_json."""
        config = ImageDatasetConfig(
            name="test",
            path="/path/to/images",
        )
        json_str = config.to_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["name"], "test")

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "name": "test",
            "dataset_type": "image",
            "path": "/path",
            "image_size": [224, 224],
        }
        config = DatasetConfig.from_dict(data)
        self.assertIsInstance(config, ImageDatasetConfig)

    def test_text_config_from_file(self):
        """Test creating TextDatasetConfig from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "name": "test",
                    "dataset_type": "text",
                    "path": "/path/to/text.jsonl",
                    "format": "jsonl",
                },
                f,
            )
            temp_path = f.name

        try:
            config = DatasetConfig.from_file(temp_path)
            self.assertIsInstance(config, TextDatasetConfig)
            self.assertEqual(config.name, "test")
        finally:
            Path(temp_path).unlink()


class TestDatasetRegistry(unittest.TestCase):
    """Test DatasetRegistry."""

    def setUp(self):
        """Set up test fixtures."""
        self.registry = create_default_registry()

        # Create sample datasets for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        # Create image directories
        (self.root / "class1").mkdir()
        self._create_test_image(self.root / "class1" / "img1.ppm")

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def _create_test_image(self, path: Path):
        """Create a simple test PPM image."""
        with open(path, "wb") as f:
            f.write(b"P6\n32 32\n255\n")
            for _ in range(32 * 32):
                f.write(bytes([100, 150, 200]))

    def test_registry_list_datasets(self):
        """Test listing datasets."""
        datasets = self.registry.list_datasets()
        self.assertIsInstance(datasets, list)

    def test_registry_register_dataset(self):
        """Test registering a dataset."""
        config = ImageDatasetConfig(
            name="test_img",
            path=str(self.root),
        )
        dataset = ImageDataset.from_config(config)
        self.registry.register_dataset("test_img", dataset, config)

        retrieved = self.registry.get_dataset("test_img")
        self.assertIsNotNone(retrieved)

    def test_registry_load_from_config(self):
        """Test loading dataset from config."""
        config = ImageDatasetConfig(
            name="test_img",
            path=str(self.root),
        )
        dataset = self.registry.load_from_config(config)
        self.assertIsInstance(dataset, ImageDataset)

    def test_registry_create_dataloader(self):
        """Test creating a DataLoader through registry."""
        config = ImageDatasetConfig(
            name="test_img",
            path=str(self.root),
        )
        dataset = self.registry.load_from_config(config)
        loader = self.registry.create_dataloader(dataset, batch_size=1)
        self.assertIsInstance(loader, DataLoader)

    def test_registry_remove_dataset(self):
        """Test removing a dataset."""
        config = ImageDatasetConfig(
            name="test_img",
            path=str(self.root),
        )
        dataset = ImageDataset.from_config(config)
        self.registry.register_dataset("test_img", dataset)

        removed = self.registry.remove_dataset("test_img")
        self.assertTrue(removed)
        self.assertIsNone(self.registry.get_dataset("test_img"))


class TestIntegration(unittest.TestCase):
    """Integration tests for complete data pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        # Create image directories
        (self.root / "cat").mkdir()
        (self.root / "dog").mkdir()
        self._create_test_image(self.root / "cat" / "cat1.ppm")
        self._create_test_image(self.root / "dog" / "dog1.ppm")

        # Create text file
        self.text_file = self.root / "text.jsonl"
        with open(self.text_file, "w") as f:
            f.write('{"prompt": "hello", "label": "pos"}\n')
            f.write('{"prompt": "goodbye", "label": "neg"}\n')

        # Create tabular file
        self.csv_file = self.root / "data.csv"
        with open(self.csv_file, "w") as f:
            f.write("f1,f2,label\n")
            f.write("0.1,0.2,A\n")
            f.write("0.3,0.4,B\n")

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def _create_test_image(self, path: Path):
        """Create a simple test PPM image."""
        with open(path, "wb") as f:
            f.write(b"P6\n32 32\n255\n")
            for _ in range(32 * 32):
                f.write(bytes([100, 150, 200]))

    def test_load_all_dataset_types(self):
        """Test loading all dataset types."""
        registry = create_default_registry()

        # Load image dataset
        img_config = ImageDatasetConfig(name="images", path=str(self.root / "cat"))
        img_dataset = registry.load_from_config(img_config)
        self.assertIsInstance(img_dataset, ImageDataset)

        # Load text dataset
        text_config = TextDatasetConfig(name="text", path=str(self.text_file))
        text_dataset = registry.load_from_config(text_config)
        self.assertIsInstance(text_dataset, TextDataset)

        # Load tabular dataset
        tab_config = TabularDatasetConfig(
            name="tabular",
            path=str(self.csv_file),
            label_column="label",
        )
        tab_dataset = registry.load_from_config(tab_config)
        self.assertIsInstance(tab_dataset, TabularDataset)

    def test_batch_iteration_all_types(self):
        """Test iterating batches for all dataset types."""
        registry = create_default_registry()

        # Image batches
        img_config = ImageDatasetConfig(name="images", path=str(self.root / "cat"))
        img_dataset = registry.load_from_config(img_config)
        img_loader = registry.create_dataloader(img_dataset, batch_size=1)
        for images, _labels in img_loader:
            self.assertEqual(images.dim(), 4)  # (B, C, H, W)
            break

        # Text batches
        text_config = TextDatasetConfig(name="text", path=str(self.text_file))
        text_dataset = registry.load_from_config(text_config)
        text_loader = registry.create_dataloader(text_dataset, batch_size=1)
        for tokens, _labels in text_loader:
            self.assertEqual(tokens.dim(), 2)  # (B, seq_len)
            break

        # Tabular batches
        tab_config = TabularDatasetConfig(
            name="tabular",
            path=str(self.csv_file),
            label_column="label",
        )
        tab_dataset = registry.load_from_config(tab_config)
        tab_loader = registry.create_dataloader(tab_dataset, batch_size=1)
        for features, _labels in tab_loader:
            self.assertEqual(features.dim(), 2)  # (B, num_features)
            break


if __name__ == "__main__":
    unittest.main()
