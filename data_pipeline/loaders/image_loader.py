"""Image dataset loader for folder-based image datasets."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from data_pipeline.base import BaseDataset, Compose
from data_pipeline.configs import ImageDatasetConfig
from data_pipeline.transforms import ImageNormalize, ImageResize


class ImageDataset(BaseDataset):
    """PyTorch Dataset for loading images from a folder structure.

    Expects folder structure:
    ```
    root_dir/
        class1/
            image1.jpg
            image2.jpg
        class2/
            image3.jpg
    ```
    """

    def __init__(
        self,
        root_dir: str | Path,
        name: str = "image_dataset",
        description: str = "",
        image_size: tuple[int, int] = (224, 224),
        normalize: bool = True,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        transforms: Compose | None = None,
    ) -> None:
        """Initialize image dataset.

        Args:
            root_dir: Root directory containing class folders with images
            name: Dataset name
            description: Dataset description
            image_size: Target image size (H, W)
            normalize: Whether to normalize images
            mean: Normalization mean per channel
            std: Normalization std per channel
            transforms: Optional custom transforms
        """
        super().__init__(name, description)
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory not found: {root_dir}")

        self.image_size = image_size
        self.class_to_idx: dict[str, int] = {}
        self.idx_to_class: dict[int, str] = {}

        # Discover classes from subdirectories
        self._samples: list[tuple[Path, int]] = []
        class_idx = 0
        for class_dir in sorted(self.root_dir.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                self.class_to_idx[class_name] = class_idx
                self.idx_to_class[class_idx] = class_name

                # Find all image files
                for image_path in sorted(class_dir.glob("*")):
                    if image_path.suffix.lower() in [
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".gif",
                        ".bmp",
                        ".ppm",
                    ]:
                        self._samples.append((image_path, class_idx))

                class_idx += 1

        if not self._samples:
            raise ValueError(f"No images found in {root_dir}")

        # Build transforms pipeline
        if transforms is None:
            transform_list = [ImageResize(image_size)]
            if normalize:
                transform_list.append(ImageNormalize(mean, std))
            self.transforms = Compose(transform_list)
        else:
            self.transforms = transforms

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Get image and label.

        Returns:
            Tuple of (image_tensor, label_index)
        """
        image_path, label = self._samples[index]

        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise OSError(f"Failed to load image {image_path}: {e}") from e

        # Convert to tensor (H, W, C) -> (C, H, W)
        import numpy as np

        image_array = np.array(image, dtype="float32").transpose(2, 0, 1)
        image_tensor = torch.from_numpy(image_array)

        # Apply transforms
        if self.transforms:
            image_tensor = self.transforms(image_tensor)

        return image_tensor, label

    def get_class_name(self, class_idx: int) -> str:
        """Get class name from index.

        Args:
            class_idx: Class index

        Returns:
            Class name
        """
        return self.idx_to_class.get(class_idx, "unknown")

    def get_class_names(self) -> list[str]:
        """Get all class names in order.

        Returns:
            List of class names
        """
        return [self.idx_to_class[i] for i in sorted(self.idx_to_class.keys())]

    @classmethod
    def from_config(cls, config: ImageDatasetConfig) -> ImageDataset:
        """Create dataset from configuration.

        Args:
            config: ImageDatasetConfig instance

        Returns:
            ImageDataset instance
        """
        return cls(
            root_dir=config.path,
            name=config.name,
            description=config.description,
            image_size=tuple(config.image_size),
            normalize=config.normalize,
            mean=config.mean,
            std=config.std,
        )
