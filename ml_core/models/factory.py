"""Model zoo registry and factory utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Generic, Literal, TypeVar

import torch
from torch import Tensor, nn

InputType = Literal["image", "text", "tabular", "sequence"]
ActivationName = Literal["relu", "gelu", "tanh", "leaky_relu", "elu"]
RNNType = Literal["gru", "lstm", "rnn"]
SequencePooling = Literal["last", "mean", "max"]
TransformerPooling = Literal["mean", "cls"]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _activation_from_name(name: ActivationName) -> nn.Module:
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "leaky_relu":
        return nn.LeakyReLU(0.2)
    if normalized == "elu":
        return nn.ELU()
    raise ValueError(f"Unsupported activation '{name}'")


def _broadcast_kernel_sizes(
    kernel_sizes: tuple[int, ...], length: int
) -> tuple[int, ...]:
    if len(kernel_sizes) == length:
        return kernel_sizes
    if len(kernel_sizes) == 1:
        return kernel_sizes * length
    raise ValueError(
        "kernel_sizes must either have a single value or match the number of conv channels"
    )


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelMetadata:
    """Metadata shared by all models in the zoo."""

    name: str
    input_type: InputType
    task: str = "classification"
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelConfig:
    """Base configuration schema for model definitions."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


default_metadata_description = (
    "Base class for registry models. Forward must be implemented by subclasses."
)


class BaseModel(nn.Module):
    """Common functionality shared across model implementations."""

    def __init__(self, metadata: ModelMetadata) -> None:
        super().__init__()
        self._metadata = metadata

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def save(self, path: str) -> None:
        payload = {
            "state_dict": self.state_dict(),
            "metadata": self.metadata.to_dict(),
        }
        torch.save(payload, path)

    def load(self, path: str, map_location: torch.device | str = "cpu") -> None:
        payload = torch.load(path, map_location=map_location)
        state_dict = (
            payload["state_dict"]
            if isinstance(payload, dict) and "state_dict" in payload
            else payload
        )
        self.load_state_dict(state_dict)

    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover - abstract method
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Configuration schemas
# ---------------------------------------------------------------------------


@dataclass
class SimpleCNNConfig(ModelConfig):
    input_channels: int = 1
    input_height: int = 28
    input_width: int = 28
    num_classes: int = 10
    conv_channels: tuple[int, ...] = (16, 32)
    kernel_sizes: tuple[int, ...] = (3,)
    activation: ActivationName = "relu"
    dropout: float = 0.25
    use_batch_norm: bool = True
    classifier_hidden: int | None = 128


@dataclass
class DenseMLPConfig(ModelConfig):
    input_dim: int = 32
    num_classes: int = 2
    hidden_layers: tuple[int, ...] = (128, 64)
    activation: ActivationName = "relu"
    dropout: float = 0.1
    use_batch_norm: bool = True


@dataclass
class SimpleRNNConfig(ModelConfig):
    input_dim: int = 32
    num_classes: int = 2
    hidden_size: int = 64
    num_layers: int = 1
    rnn_type: RNNType = "gru"
    bidirectional: bool = False
    dropout: float = 0.0
    output_pooling: SequencePooling = "last"


@dataclass
class TransformerEncoderConfig(ModelConfig):
    input_dim: int = 32
    num_classes: int = 2
    seq_len: int = 16
    model_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    activation: ActivationName = "gelu"
    pooling: TransformerPooling = "mean"


# ---------------------------------------------------------------------------
# Model implementations
# ---------------------------------------------------------------------------


class SimpleCNNClassifier(BaseModel):
    def __init__(self, config: SimpleCNNConfig) -> None:
        metadata = ModelMetadata(
            name="simple_cnn",
            input_type="image",
            description="Lightweight convolutional classifier suitable for small images.",
        )
        super().__init__(metadata)
        self.config = config

        kernels = _broadcast_kernel_sizes(
            config.kernel_sizes, len(config.conv_channels)
        )
        layers: list[nn.Module] = []
        in_channels = config.input_channels
        for out_channels, kernel_size in zip(
            config.conv_channels, kernels, strict=True
        ):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            if config.use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(_activation_from_name(config.activation))
            layers.append(nn.MaxPool2d(2))
            if config.dropout > 0:
                layers.append(nn.Dropout2d(config.dropout))
            in_channels = out_channels
        self.feature_extractor = nn.Sequential(*layers) if layers else nn.Identity()

        self._flattened_dim = self._infer_flattened_dim()
        classifier_layers: list[nn.Module] = []
        in_features = self._flattened_dim
        if config.classifier_hidden is not None:
            classifier_layers.append(nn.Linear(in_features, config.classifier_hidden))
            classifier_layers.append(_activation_from_name(config.activation))
            if config.dropout > 0:
                classifier_layers.append(nn.Dropout(config.dropout))
            in_features = config.classifier_hidden
        classifier_layers.append(nn.Linear(in_features, config.num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def _infer_flattened_dim(self) -> int:
        with torch.no_grad():
            sample = torch.zeros(
                1,
                self.config.input_channels,
                self.config.input_height,
                self.config.input_width,
            )
            features = self.feature_extractor(sample)
        return int(features.view(1, -1).size(1))

    def forward(self, x: Tensor) -> Tensor:
        features = self.feature_extractor(x)
        features = torch.flatten(features, start_dim=1)
        return self.classifier(features)


class DenseMLPClassifier(BaseModel):
    def __init__(self, config: DenseMLPConfig) -> None:
        metadata = ModelMetadata(
            name="dense_mlp",
            input_type="tabular",
            description="Configurable feed-forward MLP classifier.",
        )
        super().__init__(metadata)
        self.config = config

        layers: list[nn.Module] = []
        in_features = config.input_dim
        for hidden_size in config.hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(_activation_from_name(config.activation))
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_features = hidden_size
        layers.append(nn.Linear(in_features, config.num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class SimpleRNNClassifier(BaseModel):
    def __init__(self, config: SimpleRNNConfig) -> None:
        metadata = ModelMetadata(
            name="simple_rnn",
            input_type="text",
            description="Sequence model backed by GRU/LSTM/RNN layers.",
        )
        super().__init__(metadata)
        self.config = config

        rnn_cls_map: dict[RNNType, type[nn.Module]] = {
            "gru": nn.GRU,
            "lstm": nn.LSTM,
            "rnn": nn.RNN,
        }
        rnn_cls = rnn_cls_map[config.rnn_type]
        dropout = config.dropout if config.num_layers > 1 else 0.0
        self.rnn = rnn_cls(
            input_size=config.input_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            bidirectional=config.bidirectional,
            dropout=dropout,
            batch_first=True,
        )
        self.pooling = (
            config.output_pooling
            if config.output_pooling in {"last", "mean", "max"}
            else "last"
        )
        features = config.hidden_size * (2 if config.bidirectional else 1)
        self.classifier = nn.Linear(features, config.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        output, _ = self.rnn(x)
        if self.pooling == "mean":
            features = output.mean(dim=1)
        elif self.pooling == "max":
            features, _ = torch.max(output, dim=1)
        else:
            features = output[:, -1, :]
        return self.classifier(features)


class TransformerEncoderClassifier(BaseModel):
    def __init__(self, config: TransformerEncoderConfig) -> None:
        metadata = ModelMetadata(
            name="transformer_encoder",
            input_type="text",
            description="Transformer encoder classifier with learnable positional encodings.",
        )
        super().__init__(metadata)
        self.config = config

        self.pooling = config.pooling
        self.input_projection = nn.Linear(config.input_dim, config.model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )
        max_positions = config.seq_len + (1 if config.pooling == "cls" else 0)
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, max_positions, config.model_dim)
        )
        if config.pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.model_dim))
        else:
            self.register_parameter("cls_token", None)
        self.norm = nn.LayerNorm(config.model_dim)
        self.classifier = nn.Linear(config.model_dim, config.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        projected = self.input_projection(x)

        tokens = projected
        if self.pooling == "cls":
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            tokens = torch.cat([cls_token, projected], dim=1)

        if tokens.size(1) > self.positional_encoding.size(1):
            raise ValueError(
                "Sequence length exceeds configured maximum. Increase seq_len in the config."
            )

        positional = self.positional_encoding[:, : tokens.size(1), :]
        encoded = tokens + positional
        encoded = self.encoder(encoded)
        encoded = self.norm(encoded)

        if self.pooling == "cls":
            features = encoded[:, 0, :]
        else:
            features = encoded.mean(dim=1)
        return self.classifier(features)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


TConfig = TypeVar("TConfig", bound=ModelConfig)


@dataclass
class ModelSpec(Generic[TConfig]):
    model_cls: type[BaseModel]
    config_cls: type[TConfig]
    default_config: TConfig | None = None


class ModelRegistry:
    """Registry of available model builders."""

    def __init__(self) -> None:
        self._registry: dict[str, ModelSpec[Any]] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register(
            name="simple_cnn",
            model_cls=SimpleCNNClassifier,
            config_cls=SimpleCNNConfig,
            default_config=SimpleCNNConfig(),
        )
        self.register(
            name="simple_rnn",
            model_cls=SimpleRNNClassifier,
            config_cls=SimpleRNNConfig,
            default_config=SimpleRNNConfig(),
        )
        self.register(
            name="transformer_encoder",
            model_cls=TransformerEncoderClassifier,
            config_cls=TransformerEncoderConfig,
            default_config=TransformerEncoderConfig(),
        )
        self.register(
            name="dense_mlp",
            model_cls=DenseMLPClassifier,
            config_cls=DenseMLPConfig,
            default_config=DenseMLPConfig(),
        )

    def register(
        self,
        name: str,
        model_cls: type[BaseModel],
        config_cls: type[TConfig],
        default_config: TConfig | None = None,
    ) -> None:
        self._registry[name] = ModelSpec(
            model_cls=model_cls, config_cls=config_cls, default_config=default_config
        )

    def create(
        self,
        name: str,
        config: ModelConfig | None = None,
        **config_kwargs: Any,
    ) -> BaseModel:
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise ValueError(f"Model '{name}' not found. Available models: {available}")

        spec = self._registry[name]
        if config is not None:
            if not isinstance(config, spec.config_cls):
                raise TypeError(
                    f"Config for '{name}' must be an instance of {spec.config_cls.__name__}."
                )
            if config_kwargs:
                raise ValueError(
                    "Cannot provide config kwargs when a config object is supplied."
                )
        else:
            base_kwargs = spec.default_config.to_dict() if spec.default_config else {}
            base_kwargs.update(config_kwargs)
            config = spec.config_cls(**base_kwargs)

        return spec.model_cls(config)

    def list_models(self) -> list[str]:
        return sorted(self._registry.keys())

    def get_config_class(self, name: str) -> type[ModelConfig]:
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise ValueError(f"Model '{name}' not found. Available models: {available}")
        return self._registry[name].config_cls


# ---------------------------------------------------------------------------
# Optimizer and scheduler helpers
# ---------------------------------------------------------------------------


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = "adam",
    learning_rate: float = 0.001,
    **kwargs: Any,
) -> torch.optim.Optimizer:
    optimizer_type = optimizer_type.lower()
    if optimizer_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, **kwargs)
    if optimizer_type == "sgd":
        momentum = kwargs.pop("momentum", 0.9)
        return torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=momentum, **kwargs
        )
    if optimizer_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, **kwargs)
    raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "step",
    **kwargs: Any,
) -> Any:
    scheduler_type = scheduler_type.lower()
    if scheduler_type == "step":
        step_size = kwargs.get("step_size", 10)
        gamma = kwargs.get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    if scheduler_type == "cosine":
        t_max = kwargs.get("T_max", kwargs.get("t_max", 50))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    if scheduler_type == "plateau":
        mode = kwargs.get("mode", "min")
        patience = kwargs.get("patience", 5)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, patience=patience
        )
    raise ValueError(f"Unknown scheduler type: {scheduler_type}")


__all__ = [
    "ModelConfig",
    "ModelMetadata",
    "BaseModel",
    "SimpleCNNConfig",
    "SimpleCNNClassifier",
    "DenseMLPConfig",
    "DenseMLPClassifier",
    "SimpleRNNConfig",
    "SimpleRNNClassifier",
    "TransformerEncoderConfig",
    "TransformerEncoderClassifier",
    "ModelRegistry",
    "create_optimizer",
    "create_scheduler",
]
