"""Environment registry.

Bundles everything the experiment driver needs for a given task domain -- the
environment factory, task factory, action space, rollout heuristic, and policy
factory -- behind a name. Adding a new environment is a matter of writing the
env module and registering one ``EnvSpec`` here; the rest of the pipeline is
unchanged. See ``ARCHITECTURE.md`` for a walkthrough.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from prism.agents import (
    HeuristicPolicyAgent,
    Policy,
    SoftmaxPolicyAgent,
)
from prism.envs import bugfix, manipulation
from prism.rollout import EnvFactory
from prism.types import Action, TaskSpec


@dataclass(frozen=True)
class EnvSpec:
    """Everything the experiment driver needs to run on a given environment.

    Keeping these together behind a name makes the rest of the pipeline
    environment-agnostic: swap the registry entry and the same MC labeling,
    PRM training, and PPO policy loop run on a different task domain.
    """

    name: str
    env_factory: EnvFactory
    task_factory: Callable[..., list[TaskSpec]]
    action_space: tuple[Action, ...]
    heuristic_factory: Callable[[], Policy]
    policy_factory: Callable[..., SoftmaxPolicyAgent]


def _bugfix_policy(
    *, strength: float, temperature: float, seed: int
) -> SoftmaxPolicyAgent:
    return SoftmaxPolicyAgent.bugfix_priors(
        strength=strength,
        temperature=temperature,
        seed=seed,
    )


def _manipulation_policy(
    *, strength: float, temperature: float, seed: int
) -> SoftmaxPolicyAgent:
    # Warm-start the robot policy with a weak prior over the canonical
    # pick-and-place sequence. At strength=0 this reduces to a near-uniform
    # policy; a positive strength gives terminal reward something to latch onto
    # while leaving room for the PRM dense reward to improve on it.
    return SoftmaxPolicyAgent.manipulation_priors(
        actions=manipulation.ACTION_SPACE,
        strength=strength,
        temperature=temperature,
        seed=seed,
    )


REGISTRY: dict[str, EnvSpec] = {
    "bugfix": EnvSpec(
        name="bugfix",
        env_factory=bugfix.ToyBugFixEnv,
        task_factory=bugfix.make_tasks,
        action_space=bugfix.ACTION_SPACE,
        heuristic_factory=HeuristicPolicyAgent,
        policy_factory=_bugfix_policy,
    ),
    "manipulation": EnvSpec(
        name="manipulation",
        env_factory=manipulation.ManipulationEnv,
        task_factory=manipulation.make_tasks,
        action_space=manipulation.ACTION_SPACE,
        heuristic_factory=manipulation.HeuristicManipulationAgent,
        policy_factory=_manipulation_policy,
    ),
}


def get_env_spec(name: str) -> EnvSpec:
    if name not in REGISTRY:
        available = ", ".join(sorted(REGISTRY))
        raise ValueError(f"Unknown env '{name}'. Available: {available}")
    return REGISTRY[name]
