"""Monte-Carlo process labeling.

Estimates a process label for a (state, action) pair by replaying the prefix up
to that action and then sampling a rollout policy to completion several times;
the empirical success rate is the label. The ``*_action_space`` variants label
*every* candidate action at every visited state, yielding a contrastive dataset
for the PRM. Rollouts run in parallel with per-rollout seeded RNGs for
determinism.
"""
from __future__ import annotations

import json
import random
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from prism.agents import HeuristicPolicyAgent, Policy, sample_action
from prism.envs.bugfix import ToyBugFixEnv
from prism.rollout import EnvFactory
from prism.types import Action, LabelledStep, TaskSpec, Trajectory


def complete_from_prefix(
    task: TaskSpec,
    prefix_actions: list[Action],
    policy: Policy,
    *,
    max_steps: int = 8,
    rng: random.Random | None = None,
    env_factory: EnvFactory = ToyBugFixEnv,
) -> bool:
    """Replay a prefix, then sample policy actions until the episode ends."""

    rng = rng or random.Random()
    env = env_factory(task, max_steps)
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
    env_factory: EnvFactory = ToyBugFixEnv,
) -> LabelledStep:
    """Estimate one process label by rolling out from after the chosen action."""

    step = trajectory.steps[step_index]
    return monte_carlo_label_action(
        trajectory,
        step_index,
        step.action,
        rollout_policy,
        rollout_count=rollout_count,
        max_steps=max_steps,
        rng=rng,
        env_factory=env_factory,
    )


def monte_carlo_label_action(
    trajectory: Trajectory,
    step_index: int,
    action: Action,
    rollout_policy: Policy,
    *,
    rollout_count: int = 8,
    max_steps: int = 8,
    rng: random.Random | None = None,
    workers: int = 4,
    env_factory: EnvFactory = ToyBugFixEnv,
) -> LabelledStep:
    """Estimate a process label for any candidate action at a visited state.

    Rollouts run in parallel across `workers` threads; each gets its own
    seeded RNG so results are deterministic when `rng` is provided.
    """

    if step_index < 0 or step_index >= len(trajectory.steps):
        raise IndexError(f"step_index out of range: {step_index}")
    if rollout_count <= 0:
        raise ValueError("rollout_count must be positive")

    base_seed = rng.randint(0, 2**31) if rng is not None else random.randint(0, 2**31)
    step = trajectory.steps[step_index]
    prefix_actions = trajectory.actions[:step_index] + [action]

    def _one_rollout(rollout_index: int) -> bool:
        child_rng = random.Random(base_seed + rollout_index)
        return complete_from_prefix(
            trajectory.task,
            prefix_actions,
            rollout_policy,
            max_steps=max_steps,
            rng=child_rng,
            env_factory=env_factory,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one_rollout, i) for i in range(rollout_count)]
        successes = sum(f.result() for f in as_completed(futures))

    return LabelledStep(
        task=trajectory.task,
        state=step.state,
        action=action,
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
    env_factory: EnvFactory = ToyBugFixEnv,
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
            env_factory=env_factory,
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
    env_factory: EnvFactory = ToyBugFixEnv,
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
                    env_factory=env_factory,
                )
            )
    return labels


def label_trajectory_action_space(
    trajectory: Trajectory,
    rollout_policy: Policy | None = None,
    *,
    rollout_count: int = 8,
    max_steps: int = 8,
    seed: int = 0,
    env_factory: EnvFactory = ToyBugFixEnv,
) -> list[LabelledStep]:
    """Label every available candidate action at every visited state."""

    policy = rollout_policy or HeuristicPolicyAgent()
    rng = random.Random(seed)
    actions = env_factory(trajectory.task, max_steps).available_actions()
    labels: list[LabelledStep] = []
    for step in trajectory.steps:
        for action in actions:
            labels.append(
                monte_carlo_label_action(
                    trajectory,
                    step.index,
                    action,
                    policy,
                    rollout_count=rollout_count,
                    max_steps=max_steps,
                    rng=rng,
                    env_factory=env_factory,
                )
            )
    return labels


def label_trajectories_action_space(
    trajectories: Iterable[Trajectory],
    rollout_policy: Policy | None = None,
    *,
    rollout_count: int = 8,
    max_steps: int = 8,
    seed: int = 0,
    env_factory: EnvFactory = ToyBugFixEnv,
) -> list[LabelledStep]:
    """Build a contrastive PRM dataset over all candidate actions per state."""

    policy = rollout_policy or HeuristicPolicyAgent()
    rng = random.Random(seed)
    labels: list[LabelledStep] = []
    for trajectory in trajectories:
        actions = env_factory(trajectory.task, max_steps).available_actions()
        for step in trajectory.steps:
            for action in actions:
                labels.append(
                    monte_carlo_label_action(
                        trajectory,
                        step.index,
                        action,
                        policy,
                        rollout_count=rollout_count,
                        max_steps=max_steps,
                        rng=rng,
                        env_factory=env_factory,
                    )
                )
    return labels


def save_labels(labels: Iterable[LabelledStep], path: str | Path) -> None:
    lines = [json.dumps(label.to_dict()) for label in labels]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_labels(path: str | Path) -> list[LabelledStep]:
    raw = Path(path).read_text(encoding="utf-8").splitlines()
    return [LabelledStep.from_dict(json.loads(line)) for line in raw if line.strip()]
