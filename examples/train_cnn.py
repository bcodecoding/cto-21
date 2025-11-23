#!/usr/bin/env python3
"""Worked example that exercises the ml_core training engine with a CNN."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml_core.training import AccuracyMetric, Trainer, TrainingConfig


class SyntheticShapesDataset(Dataset):
    """Tiny synthetic dataset made of vertical vs horizontal bars."""

    def __init__(self, num_samples: int = 1200, image_size: int = 16, seed: int = 7) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.data = torch.zeros(num_samples, 1, image_size, image_size)
        self.targets = torch.zeros(num_samples, dtype=torch.long)

        for idx in range(num_samples):
            label = torch.randint(0, 2, (1,), generator=generator).item()
            canvas = torch.zeros(1, image_size, image_size)
            if label == 0:
                canvas[:, image_size // 3 : image_size // 3 + 2, :] = 1.0
            else:
                canvas[:, :, image_size // 3 : image_size // 3 + 2] = 1.0
            noise = torch.randn_like(canvas, generator=generator) * 0.1
            self.data[idx] = canvas + noise
            self.targets[idx] = label

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.data.size(0)

    def __getitem__(self, index: int):  # pragma: no cover - trivial
        return self.data[index], self.targets[index]


class TinyCNN(nn.Module):
    """Very small CNN suitable for the synthetic dataset."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x.flatten(1))


def build_loaders(cfg: TrainingConfig):
    dataset = SyntheticShapesDataset()
    val_len = max(1, int(0.2 * len(dataset)))
    train_len = len(dataset) - val_len
    generator = torch.Generator().manual_seed(123)
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    return train_loader, val_loader


def build_model(_: TrainingConfig) -> nn.Module:
    return TinyCNN()


def build_optimizer(model: nn.Module, cfg: TrainingConfig) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)


def build_scheduler(optimizer: torch.optim.Optimizer, _: TrainingConfig):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.6)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("examples.train_cnn")

    artifact_root = project_root / "artifacts" / "cnn_example"
    artifact_root.mkdir(parents=True, exist_ok=True)

    config = TrainingConfig(
        epochs=5,
        batch_size=32,
        learning_rate=5e-3,
        checkpoint_interval=2,
        device="auto",
        run_name="cnn_example_run",
        artifact_dir=artifact_root,
    )

    trainer = Trainer(
        config=config,
        dataset_loader=build_loaders,
        model_factory=build_model,
        optimizer_factory=build_optimizer,
        loss_fn=nn.CrossEntropyLoss(),
        scheduler_factory=build_scheduler,
        metrics={"accuracy": AccuracyMetric()},
        logger=logger,
    )

    logger.info("Starting CNN training example...")
    result = trainer.train()

    logger.info("Training finished! Artifacts available at %s", result.artifact_dir)
    logger.info("Trace file: %s", result.trace_path)
    logger.info("Metrics file: %s", result.metrics_path)
    logger.info("Checkpoints: %s", [str(path) for path in result.checkpoints])

    logger.info("Per-epoch summary:")
    for epoch_result in result.history:
        logger.info("Epoch %d - train: %s", epoch_result.epoch, epoch_result.train_metrics)
        if epoch_result.val_metrics:
            logger.info("            val:   %s", epoch_result.val_metrics)

    logger.info("Example completed successfully. Inspect artifacts for saved models and metrics.")


if __name__ == "__main__":
    main()
