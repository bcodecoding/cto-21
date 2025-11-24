"""Unit tests for the model zoo registry and factory."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from ml_core.models import (
    BaseModel,
    DenseMLPClassifier,
    DenseMLPConfig,
    ModelRegistry,
    SimpleCNNClassifier,
    SimpleCNNConfig,
    SimpleRNNClassifier,
    SimpleRNNConfig,
    TransformerEncoderClassifier,
    TransformerEncoderConfig,
)


class TestModelRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = ModelRegistry()

    def test_list_models(self):
        models = self.registry.list_models()
        expected = ["dense_mlp", "simple_cnn", "simple_rnn", "transformer_encoder"]
        self.assertEqual(models, expected)

    def test_create_simple_cnn_with_defaults(self):
        model = self.registry.create("simple_cnn")
        self.assertIsInstance(model, SimpleCNNClassifier)
        self.assertIsInstance(model, BaseModel)
        self.assertEqual(model.metadata.name, "simple_cnn")
        self.assertEqual(model.metadata.input_type, "image")

    def test_create_simple_rnn_with_defaults(self):
        model = self.registry.create("simple_rnn")
        self.assertIsInstance(model, SimpleRNNClassifier)
        self.assertIsInstance(model, BaseModel)
        self.assertEqual(model.metadata.name, "simple_rnn")
        self.assertEqual(model.metadata.input_type, "text")

    def test_create_transformer_with_defaults(self):
        model = self.registry.create("transformer_encoder")
        self.assertIsInstance(model, TransformerEncoderClassifier)
        self.assertIsInstance(model, BaseModel)
        self.assertEqual(model.metadata.name, "transformer_encoder")
        self.assertEqual(model.metadata.input_type, "text")

    def test_create_dense_mlp_with_defaults(self):
        model = self.registry.create("dense_mlp")
        self.assertIsInstance(model, DenseMLPClassifier)
        self.assertIsInstance(model, BaseModel)
        self.assertEqual(model.metadata.name, "dense_mlp")
        self.assertEqual(model.metadata.input_type, "tabular")

    def test_create_with_kwargs(self):
        model = self.registry.create("dense_mlp", input_dim=64, num_classes=5)
        self.assertEqual(model.config.input_dim, 64)
        self.assertEqual(model.config.num_classes, 5)

    def test_create_with_config_object(self):
        config = DenseMLPConfig(
            input_dim=100, num_classes=3, hidden_layers=(256, 128, 64)
        )
        model = self.registry.create("dense_mlp", config=config)
        self.assertEqual(model.config.input_dim, 100)
        self.assertEqual(model.config.num_classes, 3)
        self.assertEqual(model.config.hidden_layers, (256, 128, 64))

    def test_create_with_config_and_kwargs_raises_error(self):
        config = DenseMLPConfig()
        with self.assertRaises(ValueError):
            self.registry.create("dense_mlp", config=config, input_dim=50)

    def test_create_unknown_model_raises_error(self):
        with self.assertRaises(ValueError) as ctx:
            self.registry.create("unknown_model")
        self.assertIn("unknown_model", str(ctx.exception))
        self.assertIn("Available models", str(ctx.exception))

    def test_get_config_class(self):
        config_cls = self.registry.get_config_class("dense_mlp")
        self.assertEqual(config_cls, DenseMLPConfig)

    def test_get_config_class_unknown_model(self):
        with self.assertRaises(ValueError):
            self.registry.get_config_class("unknown_model")


class TestSimpleCNN(unittest.TestCase):
    def test_forward_shape(self):
        config = SimpleCNNConfig(
            input_channels=1,
            input_height=28,
            input_width=28,
            num_classes=10,
            conv_channels=(16, 32),
            kernel_sizes=(3,),
            dropout=0.0,
        )
        model = SimpleCNNClassifier(config)
        batch_size = 4
        x = torch.randn(batch_size, 1, 28, 28)
        output = model(x)
        self.assertEqual(output.shape, (batch_size, 10))

    def test_forward_different_input_size(self):
        config = SimpleCNNConfig(
            input_channels=3,
            input_height=32,
            input_width=32,
            num_classes=5,
            conv_channels=(8, 16, 32),
            kernel_sizes=(5, 3, 3),
        )
        model = SimpleCNNClassifier(config)
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        self.assertEqual(output.shape, (2, 5))

    def test_no_classifier_hidden(self):
        config = SimpleCNNConfig(
            input_channels=1,
            input_height=16,
            input_width=16,
            num_classes=2,
            conv_channels=(8,),
            classifier_hidden=None,
        )
        model = SimpleCNNClassifier(config)
        x = torch.randn(1, 1, 16, 16)
        output = model(x)
        self.assertEqual(output.shape, (1, 2))

    def test_metadata(self):
        config = SimpleCNNConfig()
        model = SimpleCNNClassifier(config)
        self.assertEqual(model.metadata.name, "simple_cnn")
        self.assertEqual(model.metadata.input_type, "image")
        self.assertEqual(model.metadata.task, "classification")

    def test_gradient_flow(self):
        config = SimpleCNNConfig(num_classes=2)
        model = SimpleCNNClassifier(config)
        x = torch.randn(4, 1, 28, 28)
        output = model(x)
        loss = output.sum()
        loss.backward()
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)


class TestDenseMLP(unittest.TestCase):
    def test_forward_shape(self):
        config = DenseMLPConfig(input_dim=32, num_classes=2, hidden_layers=(64, 32))
        model = DenseMLPClassifier(config)
        x = torch.randn(8, 32)
        output = model(x)
        self.assertEqual(output.shape, (8, 2))

    def test_forward_single_hidden_layer(self):
        config = DenseMLPConfig(input_dim=10, num_classes=5, hidden_layers=(20,))
        model = DenseMLPClassifier(config)
        x = torch.randn(3, 10)
        output = model(x)
        self.assertEqual(output.shape, (3, 5))

    def test_forward_no_dropout(self):
        config = DenseMLPConfig(input_dim=16, num_classes=3, dropout=0.0)
        model = DenseMLPClassifier(config)
        x = torch.randn(5, 16)
        output = model(x)
        self.assertEqual(output.shape, (5, 3))

    def test_metadata(self):
        config = DenseMLPConfig()
        model = DenseMLPClassifier(config)
        self.assertEqual(model.metadata.name, "dense_mlp")
        self.assertEqual(model.metadata.input_type, "tabular")

    def test_gradient_flow(self):
        config = DenseMLPConfig(input_dim=10, num_classes=2)
        model = DenseMLPClassifier(config)
        x = torch.randn(4, 10)
        output = model(x)
        loss = output.sum()
        loss.backward()
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)


class TestSimpleRNN(unittest.TestCase):
    def test_forward_gru_last_pooling(self):
        config = SimpleRNNConfig(
            input_dim=16,
            num_classes=3,
            hidden_size=32,
            num_layers=1,
            rnn_type="gru",
            output_pooling="last",
        )
        model = SimpleRNNClassifier(config)
        x = torch.randn(4, 10, 16)
        output = model(x)
        self.assertEqual(output.shape, (4, 3))

    def test_forward_lstm_mean_pooling(self):
        config = SimpleRNNConfig(
            input_dim=20,
            num_classes=5,
            hidden_size=64,
            rnn_type="lstm",
            output_pooling="mean",
        )
        model = SimpleRNNClassifier(config)
        x = torch.randn(2, 15, 20)
        output = model(x)
        self.assertEqual(output.shape, (2, 5))

    def test_forward_rnn_max_pooling(self):
        config = SimpleRNNConfig(
            input_dim=8,
            num_classes=2,
            hidden_size=16,
            rnn_type="rnn",
            output_pooling="max",
        )
        model = SimpleRNNClassifier(config)
        x = torch.randn(3, 20, 8)
        output = model(x)
        self.assertEqual(output.shape, (3, 2))

    def test_forward_bidirectional(self):
        config = SimpleRNNConfig(
            input_dim=10,
            num_classes=4,
            hidden_size=24,
            bidirectional=True,
        )
        model = SimpleRNNClassifier(config)
        x = torch.randn(5, 12, 10)
        output = model(x)
        self.assertEqual(output.shape, (5, 4))

    def test_forward_multiple_layers(self):
        config = SimpleRNNConfig(
            input_dim=15,
            num_classes=2,
            hidden_size=30,
            num_layers=2,
            dropout=0.1,
        )
        model = SimpleRNNClassifier(config)
        x = torch.randn(6, 8, 15)
        output = model(x)
        self.assertEqual(output.shape, (6, 2))

    def test_metadata(self):
        config = SimpleRNNConfig()
        model = SimpleRNNClassifier(config)
        self.assertEqual(model.metadata.name, "simple_rnn")
        self.assertEqual(model.metadata.input_type, "text")

    def test_gradient_flow(self):
        config = SimpleRNNConfig(input_dim=10, num_classes=2, hidden_size=16)
        model = SimpleRNNClassifier(config)
        x = torch.randn(4, 8, 10)
        output = model(x)
        loss = output.sum()
        loss.backward()
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)


class TestTransformerEncoder(unittest.TestCase):
    def test_forward_mean_pooling(self):
        config = TransformerEncoderConfig(
            input_dim=16,
            num_classes=3,
            seq_len=10,
            model_dim=32,
            num_heads=2,
            num_layers=1,
            pooling="mean",
        )
        model = TransformerEncoderClassifier(config)
        x = torch.randn(4, 10, 16)
        output = model(x)
        self.assertEqual(output.shape, (4, 3))

    def test_forward_cls_pooling(self):
        config = TransformerEncoderConfig(
            input_dim=20,
            num_classes=5,
            seq_len=12,
            model_dim=64,
            num_heads=4,
            num_layers=2,
            pooling="cls",
        )
        model = TransformerEncoderClassifier(config)
        x = torch.randn(2, 12, 20)
        output = model(x)
        self.assertEqual(output.shape, (2, 5))

    def test_forward_with_dropout(self):
        config = TransformerEncoderConfig(
            input_dim=10,
            num_classes=2,
            seq_len=8,
            model_dim=24,
            num_heads=2,
            dropout=0.2,
        )
        model = TransformerEncoderClassifier(config)
        x = torch.randn(3, 8, 10)
        output = model(x)
        self.assertEqual(output.shape, (3, 2))

    def test_seq_len_exceeded_raises_error(self):
        config = TransformerEncoderConfig(
            input_dim=10,
            num_classes=2,
            seq_len=5,
            model_dim=16,
            num_heads=2,
        )
        model = TransformerEncoderClassifier(config)
        x = torch.randn(2, 10, 10)
        with self.assertRaises(ValueError) as ctx:
            model(x)
        self.assertIn("Sequence length exceeds", str(ctx.exception))

    def test_metadata(self):
        config = TransformerEncoderConfig()
        model = TransformerEncoderClassifier(config)
        self.assertEqual(model.metadata.name, "transformer_encoder")
        self.assertEqual(model.metadata.input_type, "text")

    def test_gradient_flow(self):
        config = TransformerEncoderConfig(
            input_dim=8, num_classes=2, seq_len=6, model_dim=16, num_heads=2
        )
        model = TransformerEncoderClassifier(config)
        x = torch.randn(2, 6, 8)
        output = model(x)
        loss = output.sum()
        loss.backward()
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)


class TestModelSerialization(unittest.TestCase):
    def test_save_and_load_cnn(self):
        config = SimpleCNNConfig(num_classes=3)
        model = SimpleCNNClassifier(config)

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            model.save(str(path))
            self.assertTrue(path.exists())

            model_loaded = SimpleCNNClassifier(config)
            model_loaded.load(str(path))

            x = torch.randn(2, 1, 28, 28)
            output1 = model(x)
            output2 = model_loaded(x)
            self.assertTrue(torch.allclose(output1, output2))

    def test_save_and_load_mlp(self):
        config = DenseMLPConfig(input_dim=20, num_classes=4)
        model = DenseMLPClassifier(config)

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            model.save(str(path))

            model_loaded = DenseMLPClassifier(config)
            model_loaded.load(str(path))

            x = torch.randn(3, 20)
            output1 = model(x)
            output2 = model_loaded(x)
            self.assertTrue(torch.allclose(output1, output2))

    def test_save_and_load_rnn(self):
        config = SimpleRNNConfig(input_dim=10, num_classes=2, hidden_size=16)
        model = SimpleRNNClassifier(config)

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            model.save(str(path))

            model_loaded = SimpleRNNClassifier(config)
            model_loaded.load(str(path))

            x = torch.randn(2, 8, 10)
            output1 = model(x)
            output2 = model_loaded(x)
            self.assertTrue(torch.allclose(output1, output2))

    def test_save_and_load_transformer(self):
        config = TransformerEncoderConfig(
            input_dim=8, num_classes=2, seq_len=6, model_dim=16, num_heads=2
        )
        model = TransformerEncoderClassifier(config)

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            model.save(str(path))

            model_loaded = TransformerEncoderClassifier(config)
            model_loaded.load(str(path))

            x = torch.randn(2, 6, 8)
            output1 = model(x)
            output2 = model_loaded(x)
            self.assertTrue(torch.allclose(output1, output2))


class TestModelTrainability(unittest.TestCase):
    def test_simple_cnn_training_step(self):
        config = SimpleCNNConfig(num_classes=2)
        model = SimpleCNNClassifier(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        x = torch.randn(4, 1, 28, 28)
        y = torch.randint(0, 2, (4,))

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        self.assertTrue(loss.item() > 0)

    def test_dense_mlp_training_step(self):
        config = DenseMLPConfig(input_dim=16, num_classes=3)
        model = DenseMLPClassifier(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        x = torch.randn(8, 16)
        y = torch.randint(0, 3, (8,))

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        self.assertTrue(loss.item() > 0)

    def test_simple_rnn_training_step(self):
        config = SimpleRNNConfig(input_dim=10, num_classes=2, hidden_size=16)
        model = SimpleRNNClassifier(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        x = torch.randn(4, 8, 10)
        y = torch.randint(0, 2, (4,))

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        self.assertTrue(loss.item() > 0)

    def test_transformer_training_step(self):
        config = TransformerEncoderConfig(
            input_dim=8, num_classes=2, seq_len=6, model_dim=16, num_heads=2
        )
        model = TransformerEncoderClassifier(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        x = torch.randn(4, 6, 8)
        y = torch.randint(0, 2, (4,))

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        self.assertTrue(loss.item() > 0)


if __name__ == "__main__":
    unittest.main()
