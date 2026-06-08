from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from prism.agents import SoftmaxPolicyAgent
from prism.mc_labeler import label_trajectories_action_space, save_labels
from prism.prm_model import ProcessRewardModel
from prism.prm_trainer import PRMTrainingConfig, save_training_metrics, train_prm
from prism.registry import REGISTRY, EnvSpec, get_env_spec
from prism.rollout import EnvFactory, collect_trajectories, run_episode, save_trajectories
from prism.types import Action, Step, TaskSpec


@dataclass(frozen=True)
class ExperimentConfig:
    output_dir: str = "runs/prism"
    env: str = "bugfix"
    seed: int = 7
    train_task_count: int = 30
    eval_task_count: int = 20
    max_steps: int = 8
    label_episodes_per_task: int = 3
    mc_rollouts: int = 8
    prm_epochs: int = 40
    policy_batches: int = 40
    policy_batch_size: int = 20
    eval_every: int = 2
    eval_episodes_per_task: int = 2
    policy_learning_rate: float = 0.035
    dense_reward_weight: float = 1.0
    terminal_reward_weight: float = 1.0
    gamma: float = 0.95
    advantage_clip: float = 2.0
    entropy_coef: float = 0.04
    initial_prior_strength: float = 0.0
    policy_temperature: float = 1.35


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    spec = get_env_spec(config.env)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_tasks = spec.task_factory(
        config.train_task_count, split="train", seed=config.seed
    )
    eval_tasks = spec.task_factory(
        config.eval_task_count, split="eval", seed=config.seed + 1
    )

    heuristic_trajectories = collect_trajectories(
        spec.heuristic_factory(),
        train_tasks,
        episodes_per_task=config.label_episodes_per_task,
        max_steps=config.max_steps,
        seed=config.seed + 11,
        env_factory=spec.env_factory,
    )
    exploratory_trajectories = collect_trajectories(
        spec.policy_factory(strength=0.0, temperature=2.0, seed=config.seed + 10),
        train_tasks,
        episodes_per_task=config.label_episodes_per_task,
        max_steps=config.max_steps,
        seed=config.seed + 12,
        env_factory=spec.env_factory,
    )
    label_trajectories_raw = heuristic_trajectories + exploratory_trajectories
    labels = label_trajectories_action_space(
        label_trajectories_raw,
        rollout_count=config.mc_rollouts,
        max_steps=config.max_steps,
        seed=config.seed + 13,
        env_factory=spec.env_factory,
    )
    save_trajectories(label_trajectories_raw, output_dir / "trajectories.jsonl")
    save_labels(labels, output_dir / "labels.jsonl")

    prm, prm_metrics = train_prm(
        labels,
        config=PRMTrainingConfig(
            epochs=config.prm_epochs,
            seed=config.seed + 14,
        ),
    )
    prm.save(output_dir / "prm.json")
    save_training_metrics(prm_metrics, output_dir / "prm_metrics.csv")

    baseline_policy = make_initial_policy(config, spec, seed=config.seed + 20)
    prism_policy = make_initial_policy(config, spec, seed=config.seed + 20)

    baseline_history = train_policy(
        "baseline",
        baseline_policy,
        train_tasks,
        eval_tasks,
        prm=None,
        config=config,
        env_factory=spec.env_factory,
        action_space=spec.action_space,
        seed=config.seed + 30,
    )
    prism_history = train_policy(
        "prism",
        prism_policy,
        train_tasks,
        eval_tasks,
        prm=prm,
        config=config,
        env_factory=spec.env_factory,
        action_space=spec.action_space,
        seed=config.seed + 30,
    )

    history = baseline_history + prism_history
    save_history(history, output_dir / "learning_curves.csv")
    save_svg_plot(history, output_dir / "learning_curves.svg")
    baseline_policy.save(output_dir / "baseline_policy.json")
    prism_policy.save(output_dir / "prism_policy.json")

    summary = {
        "config": asdict(config),
        "env": config.env,
        "labelled_steps": len(labels),
        "label_trajectory_success_rate": solve_rate(label_trajectories_raw),
        "final_baseline_solve_rate": baseline_history[-1]["solve_rate"],
        "final_prism_solve_rate": prism_history[-1]["solve_rate"],
        "outputs": {
            "trajectories": str(output_dir / "trajectories.jsonl"),
            "labels": str(output_dir / "labels.jsonl"),
            "prm": str(output_dir / "prm.json"),
            "curves_csv": str(output_dir / "learning_curves.csv"),
            "curves_svg": str(output_dir / "learning_curves.svg"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def make_initial_policy(
    config: ExperimentConfig, spec: EnvSpec, *, seed: int
) -> SoftmaxPolicyAgent:
    return spec.policy_factory(
        strength=config.initial_prior_strength,
        temperature=config.policy_temperature,
        seed=seed,
    )


def train_policy(
    variant: str,
    policy: SoftmaxPolicyAgent,
    train_tasks: list[TaskSpec],
    eval_tasks: list[TaskSpec],
    *,
    prm: ProcessRewardModel | None,
    config: ExperimentConfig,
    env_factory: EnvFactory,
    action_space: tuple[Action, ...],
    seed: int,
) -> list[dict[str, float | int | str]]:
    rng = random.Random(seed)
    interactions = 0
    history = [
        {
            "variant": variant,
            "batch": 0,
            "interactions": 0,
            "solve_rate": evaluate_policy(
                policy,
                eval_tasks,
                max_steps=config.max_steps,
                episodes_per_task=config.eval_episodes_per_task,
                seed=seed + 1,
                env_factory=env_factory,
            ),
        }
    ]

    for batch in range(1, config.policy_batches + 1):
        batch_items: list[tuple[Step, float]] = []
        for _ in range(config.policy_batch_size):
            task = rng.choice(train_tasks)
            trajectory = run_episode(
                policy,
                task,
                max_steps=config.max_steps,
                rng=rng,
                env_factory=env_factory,
            )
            interactions += trajectory.interactions
            rewards = step_rewards(
                trajectory.steps,
                succeeded=trajectory.succeeded,
                prm=prm,
                action_space=action_space,
                dense_weight=config.dense_reward_weight,
                terminal_weight=config.terminal_reward_weight,
                gamma=config.gamma,
            )
            returns = discounted_returns(rewards, gamma=config.gamma)
            batch_items.extend(zip(trajectory.steps, returns, strict=True))

        advantages = normalize([value for _, value in batch_items])
        for (step, _), advantage in zip(batch_items, advantages, strict=True):
            clipped = max(
                -config.advantage_clip,
                min(config.advantage_clip, advantage),
            )
            policy.update_logprob(
                step.state,
                step.action,
                scale=config.policy_learning_rate * clipped,
                old_prob=step.policy_prob,
            )
            policy.update_entropy(
                step.state,
                config.policy_learning_rate * config.entropy_coef,
            )

        if batch % config.eval_every == 0 or batch == config.policy_batches:
            history.append(
                {
                    "variant": variant,
                    "batch": batch,
                    "interactions": interactions,
                    "solve_rate": evaluate_policy(
                        policy,
                        eval_tasks,
                        max_steps=config.max_steps,
                        episodes_per_task=config.eval_episodes_per_task,
                        seed=seed + batch + 1,
                        env_factory=env_factory,
                    ),
                }
            )

    return history


def step_rewards(
    steps: list[Step],
    *,
    succeeded: bool,
    prm: ProcessRewardModel | None,
    action_space: tuple[Action, ...],
    dense_weight: float,
    terminal_weight: float,
    gamma: float,
) -> list[float]:
    """Potential-based PRM shaping + a terminal success bonus.

    The dense term is potential-based reward shaping (Ng et al. 1999):
    ``F_t = gamma * Phi(s_{t+1}) - Phi(s_t)`` with the state potential
    ``Phi(s) = max_a PRM(s, a)`` (the PRM's estimate of the best achievable
    success probability from that state).

    Using the raw PRM score as an additive reward collapses the policy: an
    action that merely scores above average at many states gets reinforced
    everywhere. PBRS instead rewards *progress* between states and provably
    leaves the optimal policy unchanged, so it cannot be reward-hacked by
    repeating a locally good-looking action.
    """
    if prm is None:
        rewards = [0.0 for _ in steps]
        if rewards and succeeded:
            rewards[-1] += terminal_weight
        return rewards

    def potential(state: str) -> float:
        return max(prm.score(state, candidate) for candidate in action_space)

    rewards = []
    for index, step in enumerate(steps):
        phi_current = potential(step.state)
        if step.done or index + 1 >= len(steps):
            phi_next = 0.0  # terminal states have zero potential
        else:
            phi_next = potential(steps[index + 1].state)
        rewards.append(dense_weight * (gamma * phi_next - phi_current))

    if rewards and succeeded:
        rewards[-1] += terminal_weight
    return rewards


def discounted_returns(rewards: list[float], *, gamma: float) -> list[float]:
    running = 0.0
    returns: list[float] = []
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()
    return returns


def normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = variance ** 0.5
    if std < 1e-8:
        return [0.0 for _ in values]
    return [(value - mean) / std for value in values]


def evaluate_policy(
    policy: SoftmaxPolicyAgent,
    tasks: list[TaskSpec],
    *,
    max_steps: int,
    episodes_per_task: int,
    seed: int,
    env_factory: EnvFactory,
) -> float:
    trajectories = collect_trajectories(
        policy,
        tasks,
        episodes_per_task=episodes_per_task,
        max_steps=max_steps,
        seed=seed,
        env_factory=env_factory,
    )
    return solve_rate(trajectories)


def solve_rate(trajectories: list[object]) -> float:
    if not trajectories:
        return 0.0
    return sum(bool(getattr(trajectory, "succeeded")) for trajectory in trajectories) / len(
        trajectories
    )


def save_history(history: list[dict[str, float | int | str]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["variant", "batch", "interactions", "solve_rate"],
        )
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def save_svg_plot(history: list[dict[str, float | int | str]], path: Path) -> None:
    width = 820
    height = 520
    left = 70
    right = 30
    top = 30
    bottom = 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_x = max(int(row["interactions"]) for row in history) or 1

    def point(row: dict[str, float | int | str]) -> tuple[float, float]:
        x = left + (int(row["interactions"]) / max_x) * plot_width
        y = top + (1.0 - float(row["solve_rate"])) * plot_height
        return x, y

    colors = {"baseline": "#d94848", "prism": "#2563eb"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#222" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#222" stroke-width="1.5"/>',
        f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="15">Environment interactions</text>',
        f'<text x="20" y="{height / 2}" transform="rotate(-90 20 {height / 2})" text-anchor="middle" font-family="Arial" font-size="15">Held-out solve rate</text>',
        f'<text x="{left}" y="{top - 10}" font-family="Arial" font-size="13">1.0</text>',
        f'<text x="{left}" y="{top + plot_height + 24}" font-family="Arial" font-size="13">0.0</text>',
        f'<text x="{left + plot_width - 30}" y="{top + plot_height + 24}" font-family="Arial" font-size="13">{max_x}</text>',
    ]

    for idx in range(6):
        value = idx / 5
        y = top + (1.0 - value) * plot_height
        lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
        )

    for variant in ("baseline", "prism"):
        rows = [row for row in history if row["variant"] == variant]
        if not rows:
            continue
        path_data = " ".join(
            f"{'M' if idx == 0 else 'L'} {x:.2f} {y:.2f}"
            for idx, row in enumerate(rows)
            for x, y in [point(row)]
        )
        lines.append(
            f'<path d="{path_data}" fill="none" stroke="{colors[variant]}" stroke-width="3"/>'
        )
        for row in rows:
            x, y = point(row)
            lines.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{colors[variant]}"/>'
            )

    lines.extend(
        [
            f'<rect x="{left + 20}" y="{top + 18}" width="150" height="58" fill="#fff" stroke="#ddd"/>',
            f'<line x1="{left + 32}" y1="{top + 38}" x2="{left + 64}" y2="{top + 38}" stroke="{colors["prism"]}" stroke-width="3"/>',
            f'<text x="{left + 72}" y="{top + 43}" font-family="Arial" font-size="14">PRISM</text>',
            f'<line x1="{left + 32}" y1="{top + 62}" x2="{left + 64}" y2="{top + 62}" stroke="{colors["baseline"]}" stroke-width="3"/>',
            f'<text x="{left + 72}" y="{top + 67}" font-family="Arial" font-size="14">Baseline</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PRISM toy RL experiment.")
    parser.add_argument("--output-dir", default="runs/prism")
    parser.add_argument(
        "--env",
        default="bugfix",
        choices=sorted(REGISTRY),
        help="Which interactive environment to run the pipeline on.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quick", action="store_true", help="Run a fast smoke experiment.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        config = ExperimentConfig(
            output_dir=args.output_dir,
            env=args.env,
            seed=args.seed,
            train_task_count=8,
            eval_task_count=5,
            label_episodes_per_task=2,
            mc_rollouts=3,
            prm_epochs=12,
            policy_batches=8,
            policy_batch_size=8,
            eval_every=2,
            eval_episodes_per_task=1,
        )
    else:
        config = ExperimentConfig(
            output_dir=args.output_dir, env=args.env, seed=args.seed
        )

    summary = run_experiment(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
