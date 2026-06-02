from __future__ import annotations

from prism.envs.bugfix import ACTION_SPACE
from prism.mc_labeler import label_trajectory
from prism.rollout import run_episode
from prism.types import Action, TaskSpec


class PlusMinusSolver:
    def action_distribution(
        self, state: str, actions: tuple[Action, ...] = ACTION_SPACE
    ) -> dict[Action, float]:
        lower = state.lower()
        if "tests=passed" in lower:
            return {Action("finish"): 1.0}
        if "patch=correct" in lower:
            return {Action("run_tests"): 1.0}
        if "return a - b" in lower:
            return {Action("edit", "fix_plus_minus"): 1.0}
        return {Action("read_file", "src/module.py"): 1.0}


def test_run_episode_records_successful_trajectory() -> None:
    trajectory = run_episode(
        PlusMinusSolver(),
        TaskSpec(task_id="unit-0001", bug_type="plus_minus"),
        max_steps=6,
    )

    assert trajectory.succeeded is True
    assert [step.action.key for step in trajectory.steps] == [
        "read_file:src/module.py",
        "edit:fix_plus_minus",
        "run_tests",
        "finish",
    ]
    assert trajectory.steps[-1].reward == 1.0
    assert trajectory.steps[-1].success is True


def test_label_trajectory_replays_prefix_through_labelled_action() -> None:
    trajectory = run_episode(
        PlusMinusSolver(),
        TaskSpec(task_id="unit-0002", bug_type="plus_minus"),
        max_steps=6,
    )

    labels = label_trajectory(
        trajectory,
        rollout_policy=PlusMinusSolver(),
        rollout_count=4,
        max_steps=6,
    )

    edit_label = labels[1]
    assert edit_label.action.key == "edit:fix_plus_minus"
    assert [action.key for action in edit_label.prefix_actions] == [
        "read_file:src/module.py",
        "edit:fix_plus_minus",
    ]
    assert edit_label.label == 1.0


def test_terminal_success_step_labels_as_one() -> None:
    trajectory = run_episode(
        PlusMinusSolver(),
        TaskSpec(task_id="unit-0003", bug_type="plus_minus"),
        max_steps=6,
    )

    labels = label_trajectory(
        trajectory,
        rollout_policy=PlusMinusSolver(),
        rollout_count=4,
        max_steps=6,
    )

    assert labels[-1].action.key == "finish"
    assert labels[-1].label == 1.0
    assert labels[-1].rollout_successes == labels[-1].rollout_count

