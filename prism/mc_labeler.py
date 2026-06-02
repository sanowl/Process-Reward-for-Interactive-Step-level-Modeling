from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path

from prism.agents import HeuristicPolicyAgent, Policy, sample_action
from prism.envs.bugfix import ToyBugFixEnv
from prism.types import Action, LabelledStep, TaskSpec, Trajectory


def complete_from_prefix(
    task: TaskSpec,
    prefix_actions: list[Action],
    policy: Policy,
    *,
    max_steps: int = 8,
    rng: random.Random | None = None,
) -> bool:
    """Replay a prefix, then sample policy actions until the episode ends."""

    rng = rng or random.Random()
    env = ToyBugFixEnv(task, max_steps=max_steps)
    env.replay(prefix_actions)

    while not env.done:
        state = env.state_text()
        distribution = policy.action_distribution(state, env.available_actions())
        action = sample_action(distribution, rng)
        env.step(action)

    return env.success


def monte_carlo_label_step(
    trajectory: Trajectory,
    step_index: int,
    rollout_policy: Policy,
    *,
    rollout_count: int = 8,
    max_steps: int = 8,
    rng: random.Random | None = None,
) -> LabelledStep:
    """Estimate one process label by rolling out from after the chosen action."""

    if step_index < 0 or step_index >= len(trajectory.steps):
        raise IndexError(f"step_index out of range: {step_index}")
    if rollout_count <= 0:
        raise ValueError("rollout_count must be positive")

    rng = rng or random.Random()
    step = trajectory.steps[step_index]
    prefix_actions = trajectory.actions[: step_index + 1]

    successes = 0
    for _ in range(rollout_count):
        if complete_from_prefix(
            trajectory.task,
            prefix_actions,
            rollout_policy,
            max_steps=max_steps,
            rng=rng,
        ):
            successes += 1

    return LabelledStep(
        task=trajectory.task,
        state=step.state,
        action=step.action,
        label=successes / rollout_count,
        rollout_successes=successes,
        rollout_count=rollout_count,
        prefix_actions=prefix_actions,
    )


def label_trajectory(
    trajectory: Trajectory,
    rollout_policy: Policy | None = None,
    *,
    rollout_count: int = 8,
    max_steps: int = 8,
    seed: int = 0,
) -> list[LabelledStep]:
    """Create Monte Carlo process labels for every step in a trajectory."""

    policy = rollout_policy or HeuristicPolicyAgent()
    rng = random.Random(seed)
    return [
        monte_carlo_label_step(
            trajectory,
            step.index,
            policy,
            rollout_count=rollout_count,
            max_steps=max_steps,
            rng=rng,
        )
        for step in trajectory.steps
    ]


def label_trajectories(
    trajectories: Iterable[Trajectory],
    rollout_policy: Policy | None = None,
    *,
    rollout_count: int = 8,
    max_steps: int = 8,
    seed: int = 0,
) -> list[LabelledStep]:
    """Label a collection of trajectories with deterministic per-run sampling."""

    policy = rollout_policy or HeuristicPolicyAgent()
    rng = random.Random(seed)
    labels: list[LabelledStep] = []
    for trajectory in trajectories:
        for step in trajectory.steps:
            labels.append(
                monte_carlo_label_step(
                    trajectory,
                    step.index,
                    policy,
                    rollout_count=rollout_count,
                    max_steps=max_steps,
                    rng=rng,
                )
            )
    return labels


def save_labels(labels: Iterable[LabelledStep], path: str | Path) -> None:
    lines = [json.dumps(label.to_dict()) for label in labels]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_labels(path: str | Path) -> list[LabelledStep]:
    raw = Path(path).read_text(encoding="utf-8").splitlines()
    return [LabelledStep.from_dict(json.loads(line)) for line in raw if line.strip()]

