# Contributing

Thanks for helping improve this research repository.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the complete fast verification suite before opening a pull request:

```bash
make check
```

The tests use Python's standard-library `unittest` runner.  The manuscript
figure manifest is checked separately so missing committed summaries fail
with an explicit path.

## Changes to experiments

- Keep random seeds and all substantive settings in the saved configuration.
- Prefer `--plot-only` when changing presentation without changing numerical
  results.
- Do not commit Python/Numba caches or resumable `*checkpoint*` files.
- Commit the smallest final summary needed to redraw a reported figure.
- Preserve historical serialized method keys when old result files depend on
  them; use `GE-betting` for new user-facing labels.

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the manuscript
workflow and [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the experiment
catalog.
