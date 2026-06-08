from __future__ import annotations

from prism.envs.manipulation import (
    ACTION_SPACE,
    SCENE_LIBRARY,
    HeuristicManipulationAgent,
    ManipulationEnv,
    make_tasks,
)
from prism.mc_labeler import label_trajectory_action_space, monte_carlo_label_action
from prism.rollout import run_episode
from prism.types import Action, TaskSpec


def _env_factory(task: TaskSpec, max_steps: int) -> ManipulationEnv:
    return ManipulationEnv(task, max_steps=max_steps)


class CubeSolver:
    """Deterministic optimal policy for the cube scenes used in tests."""

    def action_distribution(
        self, state: str, actions: tuple[Action, ...] = ACTION_SPACE
    ) -> dict[Action, float]:
        lower = state.lower()
        if "placed=yes" in lower:
            return {Action("done"): 1.0}
        if "holding=object" in lower:
            if "gripper=target" in lower:
                return {Action("release"): 1.0}
            return {Action("move_to", "target"): 1.0}
        if "gripper=object" in lower:
            return {Action("grasp_pinch"): 1.0}
        return {Action("move_to", "object"): 1.0}


def test_canonical_sequence_solves_every_scene() -> None:
    for scene_type, scene in SCENE_LIBRARY.items():
        env = ManipulationEnv(TaskSpec(task_id=scene_type, bug_type=scene_type))
        sequence = [
            Action("move_to", "object"),
            Action(scene.grasp_action),
            Action("move_to", "target"),
            Action("release"),
            Action("done"),
        ]
        for action in sequence:
            env.step(action)
        assert env.success is True, scene_type
        assert env.placed is True, scene_type


def test_wrong_grasp_does_not_secure_object() -> None:
    # cube needs grasp_pinch; suction should slip and recoil the gripper home.
    env = ManipulationEnv(TaskSpec(task_id="cube", bug_type="cube_left"))
    env.step(Action("move_to", "object"))
    env.step(Action("grasp_suction"))
    assert env.holding is False
    assert env.gripper == "home"


def test_release_outside_target_loses_object() -> None:
    env = ManipulationEnv(TaskSpec(task_id="cube", bug_type="cube_left"))
    env.step(Action("move_to", "object"))
    env.step(Action("grasp_pinch"))
    env.step(Action("release"))  # released at the pick location, not the target
    assert env.placed is False
    assert env.object_lost is True


def test_run_episode_records_successful_trajectory() -> None:
    trajectory = run_episode(
        CubeSolver(),
        TaskSpec(task_id="cube-0001", bug_type="cube_left"),
        max_steps=8,
        env_factory=_env_factory,
    )
    assert trajectory.succeeded is True
    assert [step.action.key for step in trajectory.steps] == [
        "move_to:object",
        "grasp_pinch",
        "move_to:target",
        "release",
        "done",
    ]


def test_heuristic_agent_solves_scenes() -> None:
    for task in make_tasks(5, seed=3):
        trajectory = run_episode(
            HeuristicManipulationAgent(),
            task,
            max_steps=12,
            env_factory=_env_factory,
        )
        assert trajectory.succeeded is True, task.bug_type


def test_mc_label_high_on_solution_path_low_on_bad_action() -> None:
    trajectory = run_episode(
        CubeSolver(),
        TaskSpec(task_id="cube-0002", bug_type="cube_left"),
        max_steps=8,
        env_factory=_env_factory,
    )

    # The grasp_pinch step is on the solution path: a competent rollout policy
    # should finish from there nearly always.
    good = monte_carlo_label_action(
        trajectory,
        step_index=1,
        action=Action("grasp_pinch"),
        rollout_policy=HeuristicManipulationAgent(),
        rollout_count=8,
        max_steps=8,
        env_factory=_env_factory,
    )
    # Pushing the object off the table at the same state dooms the episode.
    bad = monte_carlo_label_action(
        trajectory,
        step_index=1,
        action=Action("push"),
        rollout_policy=HeuristicManipulationAgent(),
        rollout_count=8,
        max_steps=8,
        env_factory=_env_factory,
    )

    assert good.label > bad.label
    assert bad.label == 0.0


def test_action_space_labeling_covers_every_action() -> None:
    trajectory = run_episode(
        CubeSolver(),
        TaskSpec(task_id="cube-0003", bug_type="cube_left"),
        max_steps=8,
        env_factory=_env_factory,
    )
    labels = label_trajectory_action_space(
        trajectory,
        rollout_policy=HeuristicManipulationAgent(),
        rollout_count=3,
        max_steps=8,
        env_factory=_env_factory,
    )
    assert len(labels) == len(trajectory.steps) * len(ACTION_SPACE)
