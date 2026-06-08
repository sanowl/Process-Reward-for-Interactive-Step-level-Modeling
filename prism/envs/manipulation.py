from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

from prism.agents import normalize_scores
from prism.types import Action, TaskSpec


@dataclass(frozen=True)
class SceneSpec:
    """A tabletop pick-and-place scene.

    `scene_type` is stored in TaskSpec.bug_type (reused as a generic task-variant
    field). Each object type requires a specific grasp macro, mirroring how each
    bug type in the bug-fix env requires a specific patch.
    """

    scene_type: str
    object_type: str
    grasp_action: str
    object_desc: str
    target_side: str
    success: str


# Object -> the grasp macro that actually secures it. Choosing the wrong grasp
# makes the object slip, recoiling the gripper home (a costly, recoverable
# mistake) -- the manipulation analogue of applying the wrong patch.
GRASP_FOR_OBJECT: dict[str, str] = {
    "cube": "grasp_pinch",
    "sphere": "grasp_suction",
    "bowl": "grasp_two_hand",
}


SCENE_LIBRARY: dict[str, SceneSpec] = {
    "cube_left": SceneSpec(
        scene_type="cube_left",
        object_type="cube",
        grasp_action="grasp_pinch",
        object_desc="a small wooden cube",
        target_side="left",
        success="The cube rests inside the left target zone.",
    ),
    "cube_right": SceneSpec(
        scene_type="cube_right",
        object_type="cube",
        grasp_action="grasp_pinch",
        object_desc="a small wooden cube",
        target_side="right",
        success="The cube rests inside the right target zone.",
    ),
    "sphere_left": SceneSpec(
        scene_type="sphere_left",
        object_type="sphere",
        grasp_action="grasp_suction",
        object_desc="a smooth round sphere",
        target_side="left",
        success="The sphere rests inside the left target zone.",
    ),
    "sphere_right": SceneSpec(
        scene_type="sphere_right",
        object_type="sphere",
        grasp_action="grasp_suction",
        object_desc="a smooth round sphere",
        target_side="right",
        success="The sphere rests inside the right target zone.",
    ),
    "bowl_center": SceneSpec(
        scene_type="bowl_center",
        object_type="bowl",
        grasp_action="grasp_two_hand",
        object_desc="a wide ceramic bowl",
        target_side="center",
        success="The bowl rests inside the center target zone.",
    ),
}


ACTION_SPACE: tuple[Action, ...] = (
    Action("inspect"),
    Action("move_to", "object"),
    Action("move_to", "target"),
    Action("move_to", "home"),
    Action("grasp_pinch"),
    Action("grasp_suction"),
    Action("grasp_two_hand"),
    Action("release"),
    Action("push"),
    Action("done"),
)

GRASP_ACTIONS: frozenset[str] = frozenset(GRASP_FOR_OBJECT.values())


class ManipulationEnv:
    """Small interactive tabletop pick-and-place environment.

    The environment is intentionally tiny, but it preserves the shape of
    long-horizon robot manipulation: perceive the scene, move the gripper,
    select a grasp suited to the object, transport it, and release it on target.

    The canonical solution is:
        move_to:object -> <correct grasp> -> move_to:target -> release -> done
    """

    def __init__(self, task: TaskSpec, max_steps: int = 8):
        if task.bug_type not in SCENE_LIBRARY:
            raise ValueError(f"Unknown scene_type: {task.bug_type}")
        self.task = task
        self.scene = SCENE_LIBRARY[task.bug_type]
        self.max_steps = max_steps
        self.step_count = 0
        self.gripper = "home"
        self.holding = False
        self.placed = False
        self.object_lost = False
        self.done = False
        self.success = False
        self.transcript: list[tuple[Action, str]] = []

    def available_actions(self) -> tuple[Action, ...]:
        return ACTION_SPACE

    def state_text(self) -> str:
        holding = "object" if self.holding else "none"
        placed = "yes" if self.placed else "no"
        lines = [
            f"Task: {self.task.task_id}",
            "Goal: pick up the object and place it in the target zone, then signal done.",
            (
                f"Status: steps taken: {self.step_count}; gripper={self.gripper}; "
                f"holding={holding}; placed={placed}; object={self.scene.object_type}; "
                f"target={self.scene.target_side}"
            ),
        ]
        if not self.transcript:
            lines.append("Transcript: no actions yet.")
            return "\n".join(lines)

        lines.append("Transcript:")
        for action, observation in self.transcript[-10:]:
            lines.append(f"- action={action.key}")
            lines.append(f"  observation={observation}")
        return "\n".join(lines)

    def step(self, action: Action) -> tuple[str, bool, bool]:
        if self.done:
            return "Episode already finished.", self.done, self.success

        self.step_count += 1
        observation = self._handle_action(action)

        if not self.done and self.step_count >= self.max_steps:
            self.done = True
            self.success = False
            observation = f"{observation}\nStep budget exhausted; task failed."

        self.transcript.append((action, observation))
        return observation, self.done, self.success

    def replay(self, actions: list[Action]) -> None:
        for action in actions:
            if self.done:
                break
            self.step(action)

    def _handle_action(self, action: Action) -> str:
        if action.name == "inspect":
            return (
                f"Scene: {self.scene.object_desc} sits at the pick location; "
                f"the target zone is to the {self.scene.target_side}."
            )

        if action.name == "move_to":
            if action.argument in {"object", "target", "home"}:
                self.gripper = action.argument
                carry = " carrying the object" if self.holding else ""
                return f"Gripper moved to the {action.argument} location{carry}."
            return f"Unreachable location: {action.argument}"

        if action.name in GRASP_ACTIONS:
            return self._handle_grasp(action.name)

        if action.name == "release":
            return self._handle_release()

        if action.name == "push":
            if not self.holding:
                self.object_lost = True
                return "Pushed the object; it skids off the table and is now out of reach."
            return "Cannot push while holding the object."

        if action.name == "done":
            self.done = True
            self.success = self.placed
            if self.success:
                return f"Submission accepted. {self.scene.success}"
            return "Submission rejected; the object is not in the target zone."

        return f"Unknown action: {action.key}"

    def _handle_grasp(self, grasp_name: str) -> str:
        if self.object_lost:
            return "There is nothing at the pick location to grasp."
        if self.holding:
            return "The gripper is already holding the object."
        if self.gripper != "object":
            return "The gripper is not over the object; grasp failed."
        if grasp_name == self.scene.grasp_action:
            self.holding = True
            return f"Secured the {self.scene.object_type} with {grasp_name}."
        # Wrong grasp for this object: it slips and the arm recoils home.
        self.gripper = "home"
        return (
            f"The {grasp_name} slips off the {self.scene.object_type}; "
            "the arm recoils to the home position."
        )

    def _handle_release(self) -> str:
        if not self.holding:
            return "The gripper is not holding anything to release."
        self.holding = False
        if self.gripper == "target":
            self.placed = True
            return "Released the object into the target zone."
        # Dropped somewhere other than the target: object is lost.
        self.object_lost = True
        return "Released the object outside the target zone; it is now out of reach."


def make_tasks(count: int, split: str = "train", seed: int = 0) -> list[TaskSpec]:
    scene_types = list(SCENE_LIBRARY)
    rng = random.Random(seed)
    rotated = scene_types[:]
    rng.shuffle(rotated)
    stream = itertools.cycle(rotated)
    return [
        TaskSpec(task_id=f"{split}-{idx:04d}", bug_type=next(stream), split=split)
        for idx in range(count)
    ]


class HeuristicManipulationAgent:
    """A competent rollout policy used to estimate Monte Carlo process labels.

    It reads the structured status line, selects the grasp matched to the object,
    and drives the canonical pick-and-place sequence, with a little mass left on
    other actions so rollouts stay stochastic.
    """

    def action_distribution(
        self, state: str, actions: tuple[Action, ...] = ACTION_SPACE
    ) -> dict[Action, float]:
        lower = state.lower()
        scores = {action: 0.02 for action in actions}

        def bump(key: str, amount: float) -> None:
            for action in actions:
                if action.key == key:
                    scores[action] += amount

        placed = "placed=yes" in lower
        holding = "holding=object" in lower
        over_object = "gripper=object" in lower
        over_target = "gripper=target" in lower
        correct_grasp = grasp_for_state(lower)

        if placed:
            bump("done", 5.0)
            return normalize_scores(scores)

        if holding:
            if over_target:
                bump("release", 4.0)
            else:
                bump("move_to:target", 3.0)
            return normalize_scores(scores)

        # Not holding the object yet.
        if over_object:
            if correct_grasp is not None:
                bump(correct_grasp, 4.0)
        else:
            bump("move_to:object", 3.0)
            if "no actions yet" in lower:
                bump("inspect", 0.4)

        return normalize_scores(scores)


def grasp_for_state(lower_state: str) -> str | None:
    for object_type, grasp_action in GRASP_FOR_OBJECT.items():
        if f"object={object_type}" in lower_state:
            return grasp_action
    return None
