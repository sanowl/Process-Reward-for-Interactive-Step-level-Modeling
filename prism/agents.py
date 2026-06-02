from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Protocol

from prism.envs.bugfix import ACTION_SPACE
from prism.features import feature_index, state_features
from prism.types import Action


class Policy(Protocol):
    def action_distribution(
        self, state: str, actions: tuple[Action, ...] = ACTION_SPACE
    ) -> dict[Action, float]:
        ...


def normalize_scores(scores: dict[Action, float]) -> dict[Action, float]:
    total = sum(max(value, 0.0) for value in scores.values())
    if total <= 0.0:
        uniform = 1.0 / len(scores)
        return {action: uniform for action in scores}
    return {action: max(value, 0.0) / total for action, value in scores.items()}


def sample_action(distribution: dict[Action, float], rng: random.Random) -> Action:
    threshold = rng.random()
    cumulative = 0.0
    last_action = next(iter(distribution))
    for action, probability in distribution.items():
        cumulative += probability
        last_action = action
        if threshold <= cumulative:
            return action
    return last_action


class HeuristicPolicyAgent:
    """A simple rollout policy used to estimate Monte Carlo process labels."""

    def action_distribution(
        self, state: str, actions: tuple[Action, ...] = ACTION_SPACE
    ) -> dict[Action, float]:
        lower = state.lower()
        scores = {action: 0.02 for action in actions}

        def bump(key: str, amount: float) -> None:
            for action in actions:
                if action.key == key:
                    scores[action] += amount

        bump("list_files", 0.35)
        bump("read_file:src/module.py", 0.55)
        bump("read_file:tests/test_module.py", 0.30)
        bump("search:def", 0.25)
        bump("search:assert", 0.20)
        bump("run_tests", 0.25)

        if "tests=passed" in lower or "all tests passed" in lower:
            bump("finish", 4.0)
            bump("run_tests", -0.10)
            return normalize_scores(scores)

        if "patch=correct" in lower or "patch=wrong" in lower:
            bump("run_tests", 2.0)
            bump("finish", -0.02)

        inferred_patch = infer_patch_action(state)
        if inferred_patch:
            bump(f"edit:{inferred_patch}", 3.0)
            bump("run_tests", 0.35)
        elif "tests=failed" in lower:
            bump("read_file:src/module.py", 0.80)
            bump("search:def", 0.40)

        if "steps taken: 0" in lower:
            bump("run_tests", 0.40)
            bump("finish", -0.02)

        return normalize_scores(scores)


def infer_patch_action(state: str) -> str | None:
    lower = state.lower()
    if "return a - b" in lower or "add(2, 3)" in lower:
        return "fix_plus_minus"
    if "range(len(items) + 1)" in lower or "indexerror" in lower:
        return "fix_off_by_one"
    if "price - tax" in lower or "nameerror" in lower:
        return "fix_wrong_variable"
    if "sum(values) / len(values)" in lower or "zerodivisionerror" in lower:
        return "fix_empty_case"
    if "age > 18" in lower or "18 should be adult" in lower:
        return "fix_comparison"
    return None


class SoftmaxPolicyAgent:
    """Feature-linear softmax policy with a PPO-style update hook."""

    def __init__(
        self,
        actions: tuple[Action, ...] = ACTION_SPACE,
        num_features: int = 4096,
        temperature: float = 1.0,
        seed: int = 0,
    ):
        self.actions = actions
        self.num_features = num_features
        self.temperature = max(temperature, 1e-6)
        rng = random.Random(seed)
        self.weights: list[dict[int, float]] = []
        for _ in actions:
            self.weights.append({0: rng.uniform(-0.01, 0.01)})

    @classmethod
    def bugfix_priors(
        cls,
        strength: float = 0.45,
        temperature: float = 1.0,
        seed: int = 0,
    ) -> "SoftmaxPolicyAgent":
        policy = cls(temperature=temperature, seed=seed)
        policy._add_prior("bias", "list_files", 0.20 * strength)
        policy._add_prior("bias", "read_file:src/module.py", 0.55 * strength)
        policy._add_prior("bias", "read_file:tests/test_module.py", 0.30 * strength)
        policy._add_prior("bias", "search:def", 0.20 * strength)
        policy._add_prior("bias", "search:assert", 0.15 * strength)
        policy._add_prior("bias", "run_tests", 0.35 * strength)
        policy._add_prior("bias", "finish", -1.20 * strength)

        for patch in (
            "fix_plus_minus",
            "fix_off_by_one",
            "fix_wrong_variable",
            "fix_empty_case",
            "fix_comparison",
            "random_refactor",
        ):
            policy._add_prior("bias", f"edit:{patch}", -0.20 * strength)

        policy._add_prior("kw:tests_passed", "finish", 4.00 * strength)
        policy._add_prior("kw:patch_applied", "run_tests", 1.50 * strength)
        policy._add_prior("kw:tests_failed", "read_file:src/module.py", 0.55 * strength)
        policy._add_prior("kw:tests_failed", "search:def", 0.35 * strength)

        policy._add_prior("bug:plus_minus", "edit:fix_plus_minus", 3.00 * strength)
        policy._add_prior("bug:off_by_one", "edit:fix_off_by_one", 3.00 * strength)
        policy._add_prior(
            "bug:wrong_variable", "edit:fix_wrong_variable", 3.00 * strength
        )
        policy._add_prior("bug:empty_case", "edit:fix_empty_case", 3.00 * strength)
        policy._add_prior("bug:comparison", "edit:fix_comparison", 3.00 * strength)
        return policy

    def _add_prior(self, feature_name: str, action_key: str, value: float) -> None:
        action_index = self._action_index(action_key)
        if action_index is None:
            return
        idx = feature_index(feature_name, self.num_features)
        self.weights[action_index][idx] = self.weights[action_index].get(idx, 0.0) + value

    def _action_index(self, action_key: str) -> int | None:
        for idx, action in enumerate(self.actions):
            if action.key == action_key:
                return idx
        return None

    def action_distribution(
        self, state: str, actions: tuple[Action, ...] = ACTION_SPACE
    ) -> dict[Action, float]:
        features = state_features(state, self.num_features)
        action_indices = [self.actions.index(action) for action in actions]
        logits = [self._score(features, idx) / self.temperature for idx in action_indices]
        max_logit = max(logits)
        exp_logits = [math.exp(logit - max_logit) for logit in logits]
        total = sum(exp_logits)
        return {
            action: exp_logit / total
            for action, exp_logit in zip(actions, exp_logits, strict=True)
        }

    def update_logprob(self, state: str, target: Action, scale: float) -> None:
        if target not in self.actions:
            return
        features = state_features(state, self.num_features)
        distribution = self.action_distribution(state, self.actions)
        for action_index, action in enumerate(self.actions):
            coefficient = (1.0 if action == target else 0.0) - distribution[action]
            if abs(coefficient) < 1e-12:
                continue
            update = scale * coefficient
            action_weights = self.weights[action_index]
            for feature_idx, value in features.items():
                new_value = action_weights.get(feature_idx, 0.0) + update * value
                action_weights[feature_idx] = max(min(new_value, 8.0), -8.0)

    def _score(self, features: dict[int, float], action_index: int) -> float:
        action_weights = self.weights[action_index]
        return sum(action_weights.get(idx, 0.0) * value for idx, value in features.items())

    def save(self, path: str | Path) -> None:
        raw = {
            "num_features": self.num_features,
            "temperature": self.temperature,
            "actions": [action.to_dict() for action in self.actions],
            "weights": [
                {str(idx): value for idx, value in action_weights.items()}
                for action_weights in self.weights
            ],
        }
        Path(path).write_text(json.dumps(raw, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SoftmaxPolicyAgent":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        policy = cls(
            actions=tuple(Action.from_dict(action) for action in raw["actions"]),
            num_features=int(raw["num_features"]),
            temperature=float(raw.get("temperature", 1.0)),
        )
        policy.weights = [
            {int(idx): float(value) for idx, value in action_weights.items()}
            for action_weights in raw["weights"]
        ]
        return policy
