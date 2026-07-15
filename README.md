# Betting simulations

Simulation code for comparing confidence-interval widths from the betting
procedures used by the companion `empirical_bentkus` paper repository.

## Setup

From this directory, create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
python betting.py
```

The default run diagnoses the stopping-time failure of the original additive
hedge and compares it with the corrected heat-flow hedge. It uses common
permutations across candidate means and bisection for the interval endpoints,
then writes `plots/ci_width_convergence_corrected.png`.

The older, grid-based comparison remains available as
`run_experiment(...)`. Its `M_bar_n2` curve is the legacy stopped construction;
it is retained to reproduce the failure, not as the recommended method.

Run the script from this directory because it writes figures to `plots/`.
Selected publication figures can then be copied into `../paper/plots/`.

The experiment parameters are in `run_convergence_experiment(...)` and can be
overridden from Python for faster smoke tests.
