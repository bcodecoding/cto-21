"""Transform implementations for data preprocessing."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from data_pipeline.base import Transform


class ImageResize(Transform):
    """Resize image to target size."""

    def __init__(self, size: tuple[int, int]) -> None:
        """Initialize resize transform.

        Args:
            size: Target (height, width) or (size, size)
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Resize image tensor.

        Args:
            image: Image tensor of shape (C, H, W)

        Returns:
            Resized image tensor
        """
        if image.dim() == 2:
            image = image.unsqueeze(0)
        # Add batch dimension for interpolation
        image = image.unsqueeze(0)
        resized = F.interpolate(
            image, size=self.size, mode="bilinear", align_corners=False
        )
        return resized.squeeze(0)

    def __repr__(self) -> str:
        return f"ImageResize(size={self.size})"


class ImageNormalize(Transform):
    """Normalize image using mean and std."""

    def __init__(
        self,
        mean: tuple[float, ...] = (0.485, 0.456, 0.406),
        std: tuple[float, ...] = (0.229, 0.224, 0.225),
    ) -> None:
        """Initialize normalize transform.

        Args:
            mean: Mean for each channel
            std: Std for each channel
        """
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize image tensor.

        Args:
            image: Image tensor of shape (C, H, W)

        Returns:
            Normalized image tensor
        """
        if image.dtype != torch.float32:
            image = image.float()
        # Normalize to [0, 1] if needed
        if image.max() > 1.0:
            image = image / 255.0
        return (image - self.mean) / self.std

    def __repr__(self) -> str:
        return f"ImageNormalize(mean={tuple(self.mean.squeeze())}, std={tuple(self.std.squeeze())})"


class SimpleTokenizer(Transform):
    """Simple space-based tokenizer for text."""

    def __init__(
        self, max_length: int | None = None, vocab_size: int | None = None
    ) -> None:
        """Initialize tokenizer.

        Args:
            max_length: Maximum sequence length (pad/truncate to this)
            vocab_size: Optional vocabulary size for token id mapping
        """
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.vocab: dict[str, int] = {}
        self.reverse_vocab: dict[int, str] = {}

    def build_vocab(self, texts: list[str]) -> None:
        """Build vocabulary from texts.

        Args:
            texts: List of text samples
        """
        tokens = set()
        for text in texts:
            tokens.update(text.lower().split())
        tokens = sorted(tokens)
        self.vocab = {token: idx for idx, token in enumerate(tokens)}
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}

    def __call__(self, text: str) -> torch.Tensor:
        """Tokenize text.

        Args:
            text: Input text

        Returns:
            Token tensor
        """
        tokens = text.lower().split()
        token_ids = [self.vocab.get(t, 0) for t in tokens]

        if self.max_length:
            if len(token_ids) >= self.max_length:
                token_ids = token_ids[: self.max_length]
            else:
                token_ids = token_ids + [0] * (self.max_length - len(token_ids))

        return torch.tensor(token_ids, dtype=torch.long)

    def __repr__(self) -> str:
        return f"SimpleTokenizer(max_length={self.max_length}, vocab_size={len(self.vocab)})"


class NumericScaler(Transform):
    """Scale numeric features using min-max or standardization."""

    def __init__(
        self,
        method: str = "minmax",
        feature_mins: torch.Tensor | None = None,
        feature_maxs: torch.Tensor | None = None,
        feature_means: torch.Tensor | None = None,
        feature_stds: torch.Tensor | None = None,
    ) -> None:
        """Initialize scaler.

        Args:
            method: 'minmax' or 'standard'
            feature_mins: Min values per feature (for minmax)
            feature_maxs: Max values per feature (for minmax)
            feature_means: Mean values per feature (for standard)
            feature_stds: Std values per feature (for standard)
        """
        self.method = method
        self.feature_mins = feature_mins
        self.feature_maxs = feature_maxs
        self.feature_means = feature_means
        self.feature_stds = feature_stds

    def fit(self, data: torch.Tensor) -> None:
        """Fit scaler statistics from data.

        Args:
            data: Feature tensor of shape (N, D)
        """
        if self.method == "minmax":
            self.feature_mins = data.min(dim=0).values
            self.feature_maxs = data.max(dim=0).values
        elif self.method == "standard":
            self.feature_means = data.mean(dim=0)
            self.feature_stds = data.std(dim=0)

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """Scale features.

        Args:
            features: Feature tensor

        Returns:
            Scaled features
        """
        if self.method == "minmax" and self.feature_mins is not None:
            scaled = (features - self.feature_mins) / (
                self.feature_maxs - self.feature_mins + 1e-8
            )
            return torch.clamp(scaled, 0, 1)
        elif self.method == "standard" and self.feature_means is not None:
            return (features - self.feature_means) / (self.feature_stds + 1e-8)
        return features

    def __repr__(self) -> str:
        return f"NumericScaler(method={self.method})"
