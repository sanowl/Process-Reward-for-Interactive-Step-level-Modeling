"""Core data types shared across the pipeline.

Defines the serializable records that flow between components: ``Action`` and
``TaskSpec`` (inputs), ``Step``/``Trajectory`` (rollouts), and ``LabelledStep``
(a Monte-Carlo process label). All types round-trip through ``to_dict``/
``from_dict`` so trajectories and labels can be streamed as JSONL.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Action:
    """A structured tool/action call made by the agent."""

    name: str
    argument: str = ""

    @property
    def key(self) -> str:
        return f"{self.name}:{self.argument}" if self.argument else self.name

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "argument": self.argument}

    @classmethod
    def from_dict(cls, raw: dict[str, str]) -> "Action":
        return cls(name=raw["name"], argument=raw.get("argument", ""))


@dataclass(frozen=True)
class TaskSpec:
    """A task instance for the toy interactive bug-fixing environment."""

    task_id: str
    bug_type: str
    split: str = "train"

    def to_dict(self) -> dict[str, str]:
        return {
            "task_id": self.task_id,
            "bug_type": self.bug_type,
            "split": self.split,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, str]) -> "TaskSpec":
        return cls(
            task_id=raw["task_id"],
            bug_type=raw["bug_type"],
            split=raw.get("split", "train"),
        )


@dataclass
class Step:
    """One recorded step in an agent trajectory."""

    index: int
    state: str
    action: Action
    observation: str
    done: bool
    success: bool
    policy_prob: float = 1.0
    reward: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "state": self.state,
            "action": self.action.to_dict(),
            "observation": self.observation,
            "done": self.done,
            "success": self.success,
            "policy_prob": self.policy_prob,
            "reward": self.reward,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Step":
        return cls(
            index=int(raw["index"]),
            state=raw["state"],
            action=Action.from_dict(raw["action"]),
            observation=raw["observation"],
            done=bool(raw["done"]),
            success=bool(raw["success"]),
            policy_prob=float(raw.get("policy_prob", 1.0)),
            reward=float(raw.get("reward", 0.0)),
        )


@dataclass
class Trajectory:
    """A full task attempt."""

    task: TaskSpec
    steps: list[Step] = field(default_factory=list)
    succeeded: bool = False

    @property
    def interactions(self) -> int:
        return len(self.steps)

    @property
    def actions(self) -> list[Action]:
        return [step.action for step in self.steps]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
            "succeeded": self.succeeded,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Trajectory":
        return cls(
            task=TaskSpec.from_dict(raw["task"]),
            steps=[Step.from_dict(step) for step in raw["steps"]],
            succeeded=bool(raw.get("succeeded", False)),
        )


@dataclass
class LabelledStep:
    """Monte Carlo process label for a single context/action pair."""

    task: TaskSpec
    state: str
    action: Action
    label: float
    rollout_successes: int
    rollout_count: int
    prefix_actions: list[Action]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "state": self.state,
            "action": self.action.to_dict(),
            "label": self.label,
            "rollout_successes": self.rollout_successes,
            "rollout_count": self.rollout_count,
            "prefix_actions": [action.to_dict() for action in self.prefix_actions],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LabelledStep":
        return cls(
            task=TaskSpec.from_dict(raw["task"]),
            state=raw["state"],
            action=Action.from_dict(raw["action"]),
            label=float(raw["label"]),
            rollout_successes=int(raw["rollout_successes"]),
            rollout_count=int(raw["rollout_count"]),
            prefix_actions=[
                Action.from_dict(action) for action in raw.get("prefix_actions", [])
            ],
        )

