"""Toy interactive bug-fixing environment.

A tiny tool-use loop -- inspect files, search, apply a patch, run tests, submit
-- over a small library of seeded bugs. Each bug needs a specific patch action,
so solving requires reading the state and choosing the matching fix. Preserves
the shape of long-horizon agent training while running instantly.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

from prism.types import Action, TaskSpec


@dataclass(frozen=True)
class BugSpec:
    bug_type: str
    patch_action: str
    source: str
    tests: str
    failure: str
    success: str


BUG_LIBRARY: dict[str, BugSpec] = {
    "plus_minus": BugSpec(
        bug_type="plus_minus",
        patch_action="fix_plus_minus",
        source=(
            "def add(a, b):\n"
            "    # TODO: return the sum\n"
            "    return a - b\n"
        ),
        tests="assert add(2, 3) == 5",
        failure="AssertionError: add(2, 3) returned -1 instead of 5",
        success="add handles positive integer sums.",
    ),
    "off_by_one": BugSpec(
        bug_type="off_by_one",
        patch_action="fix_off_by_one",
        source=(
            "def count_items(items):\n"
            "    total = 0\n"
            "    for _ in range(len(items) + 1):\n"
            "        total += 1\n"
            "    return total\n"
        ),
        tests="assert count_items(['a', 'b']) == 2",
        failure="IndexError: loop walks one item past the end",
        success="count_items returns the exact length.",
    ),
    "wrong_variable": BugSpec(
        bug_type="wrong_variable",
        patch_action="fix_wrong_variable",
        source=(
            "def apply_discount(price, discount):\n"
            "    return price - tax\n"
        ),
        tests="assert apply_discount(100, 15) == 85",
        failure="NameError: name 'tax' is not defined",
        success="apply_discount subtracts the discount argument.",
    ),
    "empty_case": BugSpec(
        bug_type="empty_case",
        patch_action="fix_empty_case",
        source=(
            "def average(values):\n"
            "    return sum(values) / len(values)\n"
        ),
        tests="assert average([]) == 0",
        failure="ZeroDivisionError: empty list should return 0",
        success="average handles empty lists and non-empty lists.",
    ),
    "comparison": BugSpec(
        bug_type="comparison",
        patch_action="fix_comparison",
        source=(
            "def is_adult(age):\n"
            "    return age > 18\n"
        ),
        tests="assert is_adult(18) is True",
        failure="AssertionError: 18 should be adult",
        success="is_adult treats 18 as the inclusive threshold.",
    ),
}


ACTION_SPACE: tuple[Action, ...] = (
    Action("list_files"),
    Action("read_file", "src/module.py"),
    Action("read_file", "tests/test_module.py"),
    Action("search", "def"),
    Action("search", "assert"),
    Action("edit", "fix_plus_minus"),
    Action("edit", "fix_off_by_one"),
    Action("edit", "fix_wrong_variable"),
    Action("edit", "fix_empty_case"),
    Action("edit", "fix_comparison"),
    Action("edit", "random_refactor"),
    Action("run_tests"),
    Action("finish"),
)


class ToyBugFixEnv:
    """Small interactive coding environment with tool-like actions.

    The environment is intentionally tiny, but it preserves the shape of
    long-horizon agent training: inspect state, call tools, edit, test, submit.
    """

    def __init__(self, task: TaskSpec, max_steps: int = 8):
        if task.bug_type not in BUG_LIBRARY:
            raise ValueError(f"Unknown bug_type: {task.bug_type}")
        self.task = task
        self.bug = BUG_LIBRARY[task.bug_type]
        self.max_steps = max_steps
        self.step_count = 0
        self.patch_status = "none"
        self.tests_status = "unknown"
        self.done = False
        self.success = False
        self.transcript: list[tuple[Action, str]] = []

    def available_actions(self) -> tuple[Action, ...]:
        return ACTION_SPACE

    def state_text(self) -> str:
        lines = [
            f"Task: {self.task.task_id}",
            "Goal: fix the failing Python test and submit when done.",
            f"Status: steps taken: {self.step_count}; patch={self.patch_status}; tests={self.tests_status}",
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
            return "Task already finished.", self.done, self.success

        self.step_count += 1
        observation = self._handle_action(action)

        if not self.done and self.step_count >= self.max_steps:
            self.done = True
            self.success = False
            observation = f"{observation}\nMax steps exhausted; task failed."

        self.transcript.append((action, observation))
        return observation, self.done, self.success

    def replay(self, actions: list[Action]) -> None:
        for action in actions:
            if self.done:
                break
            self.step(action)

    def _handle_action(self, action: Action) -> str:
        if action.name == "list_files":
            return "Files: src/module.py, tests/test_module.py"

        if action.name == "read_file":
            if action.argument == "src/module.py":
                return f"src/module.py\n{self.bug.source}"
            if action.argument == "tests/test_module.py":
                return f"tests/test_module.py\n{self.bug.tests}"
            return f"File not found: {action.argument}"

        if action.name == "search":
            if action.argument == "def":
                return f"src/module.py matches:\n{self.bug.source}"
            if action.argument == "assert":
                return f"tests/test_module.py matches:\n{self.bug.tests}"
            return "No matches."

        if action.name == "edit":
            if action.argument == self.bug.patch_action:
                self.patch_status = "correct"
                self.tests_status = "unknown"
                return f"Applied patch {action.argument}; source now matches the intended behavior."
            self.patch_status = "wrong"
            self.tests_status = "unknown"
            return f"Applied patch {action.argument}; this changed code but did not address the failing behavior."

        if action.name == "run_tests":
            if self.patch_status == "correct":
                self.tests_status = "passed"
                return f"All tests passed. {self.bug.success}"
            self.tests_status = "failed"
            if self.patch_status == "wrong":
                return f"Tests still fail after the patch. {self.bug.failure}"
            return f"Tests failed. {self.bug.failure}"

        if action.name == "finish":
            self.done = True
            self.success = self.patch_status == "correct" and self.tests_status == "passed"
            if self.success:
                return "Submitted solution accepted."
            return "Submitted solution rejected; tests are not passing."

        return f"Unknown action: {action.key}"


def make_tasks(count: int, split: str = "train", seed: int = 0) -> list[TaskSpec]:
    bug_types = list(BUG_LIBRARY)
    rng = random.Random(seed)
    rotated = bug_types[:]
    rng.shuffle(rotated)
    stream = itertools.cycle(rotated)
    return [
        TaskSpec(task_id=f"{split}-{idx:04d}", bug_type=next(stream), split=split)
        for idx in range(count)
    ]

