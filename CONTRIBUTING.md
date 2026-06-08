# Contributing

Thanks for your interest in PRISM! This is a small research prototype, and
contributions — new environments, cleaner baselines, better evaluation, bug
fixes — are welcome.

## Development setup

```bash
git clone https://github.com/sanowl/Process-Reward-for-Interactive-Step-level-Modeling.git
cd Process-Reward-for-Interactive-Step-level-Modeling
pip install -e .
```

There are no third-party runtime dependencies; everything uses the Python ≥ 3.10
standard library. `pytest` is optional for running the tests.

## Running the tests

```bash
pytest
```

or, without installing anything:

```bash
python -c "import tests.test_manipulation_env as t; [getattr(t,n)() for n in dir(t) if n.startswith('test_')]"
python -c "import tests.test_rollout_and_labeler as t; [getattr(t,n)() for n in dir(t) if n.startswith('test_')]"
```

The tests are fast (seconds) and fully deterministic. A quick end-to-end smoke
check of the whole pipeline:

```bash
python train.py --quick --env bugfix
python train.py --quick --env manipulation
```

## Project conventions

- **Standard library only** for runtime code. If a change needs a third-party
  dependency, please open an issue to discuss it first.
- **Determinism**: rollouts and labeling are seeded. Keep new code reproducible —
  thread an explicit `rng`/`seed` rather than using global randomness.
- **Type hints** on public functions; `from __future__ import annotations` at the
  top of each module.
- **Docstrings**: a one-paragraph module docstring and docstrings on public
  classes/functions. Explain *why*, not just *what*.
- Keep the environment-agnostic core (`rollout`, `mc_labeler`, `prm_*`,
  `experiment`) free of references to any concrete environment — go through the
  registry.

## Adding an environment

See the **"Adding a new environment"** section of [ARCHITECTURE.md](ARCHITECTURE.md).
In short: write `prism/envs/<name>.py` (env + action space + `make_tasks` +
heuristic policy), optionally add features, and register one `EnvSpec`. Please
include tests modeled on `tests/test_manipulation_env.py` (canonical-solution
solve, key failure transitions, and that MC labels rank a good action above a
clearly bad one).

## Pull requests

1. Branch from `main`.
2. Add or update tests for your change.
3. Make sure the test suite and `--quick` runs of both environments pass.
4. Describe the change and its motivation in the PR.

By contributing, you agree that your contributions are licensed under the
Apache License 2.0, the same license as the project.
