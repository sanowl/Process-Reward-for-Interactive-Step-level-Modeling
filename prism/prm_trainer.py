"""Training loop for the process reward model.

Splits labelled steps into train/validation, runs epoch-wise SGD on the PRM, and
records per-epoch train/validation MSE. ``train_prm_from_file`` wires the JSONL
labels straight through to a saved model for command-line use.
"""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

from prism.mc_labeler import load_labels
from prism.prm_model import ProcessRewardModel
from prism.types import LabelledStep


@dataclass(frozen=True)
class PRMTrainingConfig:
    num_features: int = 8192
    epochs: int = 40
    learning_rate: float = 0.08
    l2: float = 1e-6
    validation_split: float = 0.15
    seed: int = 0
    shuffle: bool = True


@dataclass(frozen=True)
class PRMTrainingMetric:
    epoch: int
    train_mse: float
    validation_mse: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "epoch": self.epoch,
            "train_mse": self.train_mse,
            "validation_mse": self.validation_mse,
        }


def train_prm(
    labels: list[LabelledStep],
    *,
    config: PRMTrainingConfig | None = None,
    model: ProcessRewardModel | None = None,
) -> tuple[ProcessRewardModel, list[PRMTrainingMetric]]:
    """Train a linear PRM on Monte Carlo step labels."""

    if not labels:
        raise ValueError("Cannot train PRM with no labels")

    config = config or PRMTrainingConfig()
    rng = random.Random(config.seed)
    model = model or ProcessRewardModel(num_features=config.num_features)

    train_labels, validation_labels = split_labels(
        labels,
        validation_split=config.validation_split,
        rng=rng,
    )
    metrics: list[PRMTrainingMetric] = []

    for epoch in range(1, config.epochs + 1):
        if config.shuffle:
            rng.shuffle(train_labels)
        total_loss = 0.0
        for label in train_labels:
            total_loss += model.update_mse(
                label.state,
                label.action,
                label.label,
                learning_rate=config.learning_rate,
                l2=config.l2,
            )

        train_mse = total_loss / len(train_labels)
        validation_mse = (
            mean_squared_error(model, validation_labels) if validation_labels else None
        )
        metrics.append(
            PRMTrainingMetric(
                epoch=epoch,
                train_mse=train_mse,
                validation_mse=validation_mse,
            )
        )

    return model, metrics


def split_labels(
    labels: list[LabelledStep],
    *,
    validation_split: float,
    rng: random.Random,
) -> tuple[list[LabelledStep], list[LabelledStep]]:
    shuffled = list(labels)
    rng.shuffle(shuffled)
    if validation_split <= 0.0:
        return shuffled, []
    if validation_split >= 1.0:
        raise ValueError("validation_split must be less than 1.0")

    validation_count = int(round(len(shuffled) * validation_split))
    validation_count = max(0, min(validation_count, len(shuffled) - 1))
    if validation_count == 0:
        return shuffled, []
    return shuffled[validation_count:], shuffled[:validation_count]


def mean_squared_error(
    model: ProcessRewardModel,
    labels: list[LabelledStep],
) -> float:
    if not labels:
        return 0.0
    total = 0.0
    for label in labels:
        error = model.score(label.state, label.action) - label.label
        total += error * error
    return total / len(labels)


def save_training_metrics(
    metrics: list[PRMTrainingMetric],
    path: str | Path,
) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_mse", "validation_mse"],
        )
        writer.writeheader()
        for metric in metrics:
            writer.writerow(metric.to_dict())


def train_prm_from_file(
    labels_path: str | Path,
    model_path: str | Path,
    *,
    metrics_path: str | Path | None = None,
    config: PRMTrainingConfig | None = None,
) -> ProcessRewardModel:
    labels = load_labels(labels_path)
    model, metrics = train_prm(labels, config=config)
    model.save(model_path)
    if metrics_path is not None:
        save_training_metrics(metrics, metrics_path)
    return model

