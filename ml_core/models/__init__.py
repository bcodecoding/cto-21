"""Model zoo with registry and factory utilities."""

from ml_core.models.factory import (
    BaseModel,
    DenseMLPClassifier,
    DenseMLPConfig,
    ModelConfig,
    ModelMetadata,
    ModelRegistry,
    SimpleCNNClassifier,
    SimpleCNNConfig,
    SimpleRNNClassifier,
    SimpleRNNConfig,
    TransformerEncoderClassifier,
    TransformerEncoderConfig,
    create_optimizer,
    create_scheduler,
)

__all__ = [
    "BaseModel",
    "ModelConfig",
    "ModelMetadata",
    "ModelRegistry",
    "SimpleCNNConfig",
    "SimpleCNNClassifier",
    "DenseMLPConfig",
    "DenseMLPClassifier",
    "SimpleRNNConfig",
    "SimpleRNNClassifier",
    "TransformerEncoderConfig",
    "TransformerEncoderClassifier",
    "create_optimizer",
    "create_scheduler",
]
