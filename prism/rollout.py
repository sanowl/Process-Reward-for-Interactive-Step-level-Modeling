from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path

from prism.agents import Policy, sample_action
from prism.envs.bugfix import ToyBugFixEnv
from prism.types import Step, TaskSpec, Trajectory


def run_episode(
    policy: Policy,
    task: TaskSpec,
    *,
    max_steps: int = 8,
    rng: random.Random | None = None,
) -> Trajectory:
    """Run one policy attempt in the interactive bug-fixing environment."""

    rng = rng or random.Random()
    env = ToyBugFixEnv(task, max_steps=max_steps)
    trajectory = Trajectory(task=task)

    while not env.done:
        state = env.state_text()
        distribution = policy.action_distribution(state, env.available_actions())
        action = sample_action(distribution, rng)
        observation, done, success = env.step(action)
        trajectory.steps.append(
            Step(
                index=len(trajectory.steps),
                state=state,
                action=action,
                observation=observation,
                done=done,
                success=success,
                policy_prob=distribution[action],
                reward=1.0 if done and success else 0.0,
            )
        )

    trajectory.succeeded = env.success
    return trajectory


def collect_trajectories(
    policy: Policy,
    tasks: Iterable[TaskSpec],
    *,
    episodes_per_task: int = 1,
    max_steps: int = 8,
    seed: int = 0,
) -> list[Trajectory]:
    """Collect full task attempts for a set of tasks."""

    rng = random.Random(seed)
    trajectories: list[Trajectory] = []
    for task in tasks:
        for _ in range(episodes_per_task):
            trajectories.append(run_episode(policy, task, max_steps=max_steps, rng=rng))
    return trajectories


def save_trajectories(trajectories: Iterable[Trajectory], path: str | Path) -> None:
    """Write trajectories as JSONL so large runs can stream cleanly."""

    lines = [json.dumps(trajectory.to_dict()) for trajectory in trajectories]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_trajectories(path: str | Path) -> list[Trajectory]:
    raw = Path(path).read_text(encoding="utf-8").splitlines()
    return [Trajectory.from_dict(json.loads(line)) for line in raw if line.strip()]

