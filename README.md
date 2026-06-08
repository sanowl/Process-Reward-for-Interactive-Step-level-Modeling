# PRISM — Process Reward for Interactive Step-level Modeling

PRISM is a small, dependency-free research prototype for training and using a
**Process Reward Model (PRM)** on *interactive, multi-step agent tasks*.

Most PRM work (e.g. *Let's Verify Step by Step*, *Math-Shepherd*) targets
single-pass reasoning such as math or one-shot code generation. PRISM explores
the same idea in the setting LLM agents actually operate in: a **tool-use loop**
where the agent observes state, takes an action, gets feedback, and repeats over
many turns. The whole pipeline runs in plain Python (standard library only) so
you can read it end to end and run it on a laptop in seconds.

## Pipeline

```
collect trajectories ──▶ Monte-Carlo step labeling ──▶ train PRM ──▶ PPO policy
   (heuristic +            (contrastive: every            (linear,      (dense PRM
    exploratory)            candidate action per state)    sigmoid)      shaping)
```

1. **Roll out** trajectories with a heuristic policy and an exploratory policy.
2. **Label each step** with Monte-Carlo rollouts: replay a prefix up to a chosen
   action, then sample the rollout policy to completion; the success rate is the
   step's process label. PRISM labels **every candidate action at every visited
   state** (`label_*_action_space`), producing a richer contrastive signal than
   labeling only the action that was taken.
3. **Train a linear PRM** (`prism/prm_model.py`) with sigmoid-MSE on those labels.
4. **Train a policy** with REINFORCE/PPO-style updates, comparing a terminal-reward
   **baseline** against **PRISM**, which adds the PRM as a dense per-step reward.

## Environments

The pipeline is environment-agnostic (see `prism/registry.py`); pick one with
`--env`:

| `--env`        | Domain                       | Solution shape                                             |
|----------------|------------------------------|-----------------------------------------------------------|
| `bugfix`       | Toy interactive bug-fixing   | `read → edit → run_tests → finish`                        |
| `manipulation` | Tabletop robot pick-and-place| `move_to:object → <grasp> → move_to:target → release → done` |

The `manipulation` environment (`prism/envs/manipulation.py`) is a
robotics-shaped task with discrete **macro-actions** (the abstraction used by
agents like SayCan/RT-2). Each object type (cube / sphere / bowl) requires a
matching grasp (`grasp_pinch` / `grasp_suction` / `grasp_two_hand`), so the agent
must select the right tool for the object — the manipulation analogue of choosing
the right fix for a bug. Both environments expose the identical interface
(`state_text`, `available_actions`, `step`, `replay`), so the MC labeler, PRM, and
policy loop are shared without change.

## Install

```bash
git clone https://github.com/sanowl/Process-Reward-for-Interactive-Step-level-Modeling.git
cd Process-Reward-for-Interactive-Step-level-Modeling
pip install -e .        # no third-party dependencies
```

Requires Python ≥ 3.10.

## Quickstart

```bash
# Fast smoke run of the full pipeline (seconds)
python train.py --quick --env bugfix
python train.py --quick --env manipulation

# Full experiment
python train.py --env manipulation --output-dir runs/robot
```

Each run writes to the output directory:

| File                   | Contents                                              |
|------------------------|-------------------------------------------------------|
| `trajectories.jsonl`   | Collected agent trajectories                          |
| `labels.jsonl`         | Monte-Carlo process labels (one per state/action)     |
| `prm.json`             | Trained process reward model                          |
| `learning_curves.csv`  | Solve rate vs. environment interactions               |
| `learning_curves.svg`  | Plotted baseline-vs-PRISM curves                       |
| `summary.json`         | Config + final metrics                                 |

## Repository layout

```
prism/
  envs/
    bugfix.py         # toy interactive bug-fixing environment
    manipulation.py   # tabletop pick-and-place environment
  types.py            # Action, TaskSpec, Step, Trajectory, LabelledStep
  features.py         # hashed sparse features for states and (state, action) pairs
  agents.py           # heuristic + linear-softmax policies (PPO clip, entropy reg)
  rollout.py          # episode rollout, env_factory injection
  mc_labeler.py       # Monte-Carlo + contrastive process labeling
  prm_model.py        # linear sigmoid process reward model
  prm_trainer.py      # PRM training loop
  registry.py         # EnvSpec registry mapping --env name -> components
train.py              # end-to-end experiment driver
tests/                # unit tests for both environments and the labeler
```

## Reward shaping

PRISM uses the PRM as **potential-based reward shaping** (Ng et al., 1999):
`F = γ·Φ(s′) − Φ(s)` with the state potential `Φ(s) = maxₐ PRM(s, a)`. Adding the
raw PRM score as a per-step reward collapses the policy — an action that merely
scores above average at many states gets reinforced everywhere. Potential-based
shaping rewards *progress between states*, provably leaves the optimal policy
unchanged, and penalizes no-progress loops. The policy update also includes
**entropy regularization** to maintain exploration on the tiny action space.

## Status & limitations

This is a **research prototype / proof of concept**, not a validated result:

- The environments are deliberately tiny (5 task variants, ≤13 discrete actions,
  perfect observations). They preserve the *shape* of long-horizon agent training
  but are not a benchmark.
- Models are feature-linear (hashed bag-of-tokens), not neural. There is no LLM or
  continuous control.
- The baseline-vs-PRISM comparison is a sandbox for the method, not evidence of an
  empirical claim. Demonstrating that PRM shaping beats outcome-only reward at
  scale would require a real environment (e.g. SWE-bench, ALFWorld, WebArena), a
  neural policy, and proper ablations.

See the inline docstrings for the algorithmic details.

## Tests

```bash
pytest            # if pytest is installed
# or, with no dependencies:
python -c "import tests.test_manipulation_env as t; [getattr(t,n)() for n in dir(t) if n.startswith('test_')]"
```

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
