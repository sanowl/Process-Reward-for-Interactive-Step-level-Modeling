# Architecture

This document explains how PRISM is put together, the math behind each stage,
and how to extend it with a new environment. For a high-level overview and
quickstart, see the [README](README.md).

## Data flow

```
                 ┌─────────────────────────────────────────────────────────┐
                 │                    prism/experiment.py                    │
                 │                  (run_experiment driver)                  │
                 └─────────────────────────────────────────────────────────┘
                                          │
   tasks ──▶ rollout.collect_trajectories ──▶ Trajectory[]                   (1)
                                          │
              mc_labeler.label_trajectories_action_space ──▶ LabelledStep[]  (2)
                                          │
                    prm_trainer.train_prm ──▶ ProcessRewardModel             (3)
                                          │
              experiment.train_policy (baseline) ─┐
              experiment.train_policy (prism)  ───┴──▶ learning_curves       (4)
```

Everything is keyed off an `EnvSpec` resolved from `prism/registry.py` by the
`--env` name, so stages (1)–(4) never reference a concrete environment.

## Stage 1 — Trajectories

`rollout.run_episode(policy, task, env_factory=...)` steps a policy through an
environment until it terminates, recording a `Step` per action (state, action,
observation, `policy_prob`, terminal flags). Environments are injected as an
`env_factory` callable `(task, max_steps) -> Environment`; any object satisfying
the `Environment` protocol (`state_text`, `available_actions`, `step`, `replay`,
`done`, `success`) works.

Two policies seed the label set: a competent `Heuristic*Agent` (mostly solves)
and an exploratory near-uniform softmax (wanders). The mix gives the PRM both
good and bad steps to learn from.

## Stage 2 — Monte-Carlo process labels

For a state `s` and candidate action `a`, the process label is an estimate of
"if I take `a` here, how likely am I to eventually succeed?":

```
label(s, a) = (1/N) · Σ_{i=1..N}  1[ rollout_i succeeds ]
```

where each rollout **replays the trajectory prefix, forces action `a`, then
samples the rollout policy to termination** (`complete_from_prefix`). Rollouts
run in parallel with per-rollout seeded RNGs, so results are deterministic given
a seed.

The `*_action_space` variants label **every** action in the action space at
every visited state, not just the action that was taken. This contrastive
dataset is what lets the PRM rank good vs. bad actions at the same state, rather
than only calibrating the chosen one.

## Stage 3 — Process reward model

`ProcessRewardModel` is a feature-linear model with a logistic head:

```
PRM(s, a) = σ( w · φ(s, a) )
```

`φ(s, a)` (`features.prm_features`) combines hashed state tokens, keyword/task
indicators, the action identity, and **state × action cross features** — the
cross features are what carry "this action is right *for this kind of state*."
Training (`prm_trainer.train_prm`) is online sigmoid-MSE SGD against the labels
with an L2 penalty and a held-out validation split.

## Stage 4 — Policy optimization & PRM shaping

`train_policy` optimizes a `SoftmaxPolicyAgent` with a REINFORCE/PPO-style
update over batches of episodes, comparing two variants:

- **baseline** — reward is terminal only (`+terminal_weight` on success).
- **prism** — terminal reward **plus** a dense PRM shaping term.

### Why potential-based shaping

A naive dense reward `PRM(s, a)` (or `PRM(s,a) − 0.5`) **collapses the policy**:
an action that scores above average at many states gets reinforced everywhere,
so the policy degenerates to repeating one action. We instead use
**potential-based reward shaping** (Ng, Harada & Russell, 1999):

```
F(s, a, s') = γ · Φ(s') − Φ(s)        with   Φ(s) = max_a PRM(s, a)
```

PBRS provably leaves the optimal policy unchanged, rewards *progress* between
states, and **penalizes no-progress loops** (repeating an action that doesn't
change the state yields `(γ−1)·Φ(s) < 0`). Terminal states have `Φ = 0`.

### Entropy regularization

On a tiny action space the policy gradient still over-commits, so each update
adds an entropy-ascent step (`SoftmaxPolicyAgent.update_entropy`) that keeps the
action distribution spread out long enough to discover the solution path. Both
variants use it, so the only difference between baseline and PRISM is the reward.

## Module map

| Module | Responsibility |
|--------|----------------|
| `prism/types.py` | `Action`, `TaskSpec`, `Step`, `Trajectory`, `LabelledStep` (+ JSONL I/O) |
| `prism/features.py` | hashed state and (state, action) features |
| `prism/envs/bugfix.py` | toy interactive bug-fixing environment + tasks |
| `prism/envs/manipulation.py` | tabletop pick-and-place environment + tasks |
| `prism/agents.py` | heuristic + linear-softmax policies (PPO clip, entropy) |
| `prism/rollout.py` | episode rollout, `Environment` protocol, `env_factory` |
| `prism/mc_labeler.py` | Monte-Carlo + contrastive labeling |
| `prism/prm_model.py` | the linear process reward model |
| `prism/prm_trainer.py` | PRM training loop + metrics |
| `prism/registry.py` | `EnvSpec` registry keyed by `--env` |
| `prism/experiment.py` | end-to-end driver, reward shaping, plotting |

## Adding a new environment

The registry is the single extension point. To add an environment named `myenv`:

1. **Write the environment** in `prism/envs/myenv.py`:
   - a class implementing the `Environment` protocol (`state_text`,
     `available_actions`, `step`, `replay`, and `done`/`success` attributes),
   - an `ACTION_SPACE: tuple[Action, ...]`,
   - a `make_tasks(count, split, seed) -> list[TaskSpec]`,
   - a heuristic rollout policy implementing `action_distribution` (it should be
     competent enough that good prefixes usually succeed — otherwise the MC
     labels carry no signal).

2. **(Optional) add features** in `features.py` for any structured fields your
   state text exposes (keyword / object indicators); start them with a prefix
   that the `prm_features` cross-feature loop picks up (`kw:`, `obj:`, …).

3. **Register an `EnvSpec`** in `prism/registry.py`:

   ```python
   REGISTRY["myenv"] = EnvSpec(
       name="myenv",
       env_factory=myenv.MyEnv,
       task_factory=myenv.make_tasks,
       action_space=myenv.ACTION_SPACE,
       heuristic_factory=myenv.MyHeuristicAgent,
       policy_factory=_myenv_policy,   # builds a SoftmaxPolicyAgent over ACTION_SPACE
   )
   ```

4. **Run it**: `python train.py --quick --env myenv`. No changes to the rollout,
   labeler, PRM, or training loop are required.

## References

- Lightman et al., *Let's Verify Step by Step* (2023) — process reward models.
- Wang et al., *Math-Shepherd* (2024) — Monte-Carlo step labeling.
- Ng, Harada & Russell, *Policy invariance under reward transformations* (1999)
  — potential-based reward shaping.
