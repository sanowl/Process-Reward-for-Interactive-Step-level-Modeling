from __future__ import annotations

import json
import math
from pathlib import Path

from prism.features import prm_features
from prism.types import Action


class ProcessRewardModel:
    """Feature-linear process reward model for scoring individual agent steps."""

    def __init__(
        self,
        *,
        num_features: int = 8192,
        weights: dict[int, float] | None = None,
    ):
        self.num_features = num_features
        self.weights: dict[int, float] = dict(weights or {})

    def score(self, state: str, action: Action) -> float:
        """Return the sigmoid score in [0, 1] for one state/action pair."""

        return sigmoid(self.logit(state, action))

    def logit(self, state: str, action: Action) -> float:
        features = prm_features(state, action, self.num_features)
        return sum(self.weights.get(idx, 0.0) * value for idx, value in features.items())

    def update_mse(
        self,
        state: str,
        action: Action,
        target: float,
        *,
        learning_rate: float,
        l2: float = 0.0,
        weight_clip: float = 12.0,
    ) -> float:
        """Apply one SGD step on sigmoid-MSE loss and return squared error."""

        target = max(0.0, min(1.0, target))
        features = prm_features(state, action, self.num_features)
        prediction = sigmoid(
            sum(self.weights.get(idx, 0.0) * value for idx, value in features.items())
        )
        error = prediction - target
        gradient_scale = error * prediction * (1.0 - prediction)

        for idx, value in features.items():
            old_weight = self.weights.get(idx, 0.0)
            gradient = gradient_scale * value + l2 * old_weight
            new_weight = old_weight - learning_rate * gradient
            self.weights[idx] = max(min(new_weight, weight_clip), -weight_clip)

        return error * error

    def save(self, path: str | Path) -> None:
        raw = {
            "num_features": self.num_features,
            "weights": {str(idx): value for idx, value in self.weights.items()},
        }
        Path(path).write_text(json.dumps(raw, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ProcessRewardModel":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            num_features=int(raw["num_features"]),
            weights={int(idx): float(value) for idx, value in raw["weights"].items()},
        )


def sigmoid(value: float) -> float:
    """Numerically stable logistic sigmoid."""

    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)

