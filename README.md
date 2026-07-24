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

The driver evaluates the following fixed-horizon procedures on six bounded
distributions:

- the original two-sided product test martingale;
- the target-recalculating STaR betting rule from
  [STaR-bets-confidence-interval](https://github.com/vvoracek/STaR-bets-confidence-interval),
  used inside the same two-sided product test;
- Probit-STaR, the buffered Gaussian-digital feedback rule proposed in the
  paper, with `b_n = n**(2/3)` and one fixed terminal uniform per arm during
  each confidence-set inversion;
- Construction 3, the original nonnegative Bentkus/heat-flow hedge;
- a target-recalculating Bentkus hedge that re-optimizes the local
  squared-hinge strike from current wealth, remaining variance, and the fixed
  rejection target;
- a matched-clock squared-hinge feedback rule and its guarded version, whose
  leverage is capped by the product-STaR square-root leverage; and
- a matched-clock target-capped quadratic feedback rule.

The common sample-size grid is
`n = 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000`;
both experiment drivers enforce `n <= 10**6`.  The current publication-scale
protocol uses 50 datasets through `n = 10**4`, 30 at `n = 5 * 10**4` and
`10**5`, and 20 at `n = 5 * 10**5` and `10**6`:

```python
import betting

num_sims = {
    10: 50,
    50: 50,
    100: 50,
    500: 50,
    1_000: 50,
    5_000: 50,
    10_000: 50,
    50_000: 30,
    100_000: 30,
    500_000: 20,
    1_000_000: 20,
}
betting.run_experiment(num_sims=num_sims)
```

`num_sims` may be either one positive integer, used at every sample size, or a
mapping with one positive count per requested `n`.  The bare
`python betting.py` invocation uses the adaptive publication protocol above,
stored as `PUBLICATION_SIMULATION_COUNTS`; an explicit
`run_experiment(num_sims=50)` requests 50 datasets at every horizon.  Both
`run_experiment(...)` and `run_dp_experiment(...)` accept this form.  Their
JSON output records the resolved mapping as `num_sims_by_n`, along with the
exact `n_values` and RNG `seed`, so unequal Monte Carlo counts remain explicit
in saved results.  The main JSON also records the capped-ramp width and its
geometric first-crossing scan settings.

The fixed-claim and target-recalculating Bentkus/heat-flow curves use the
observed chronological order.  The target-recalculating curve and the matched
squared-hinge feedback share the same Gaussian feedback map; their remaining
finite-sample differences come from the variance clock and regularization.

The target-capped rule plans against the unit-width (`eta = 1`) quadratic ramp

```text
g_eta(u) = 0                    for u <= 0,
           (u / eta)^2         for 0 < u < eta,
           1                    for u >= eta.
```

The cap removes squared-hinge payoff overshoot and makes the continuation
value closer to a digital target while preserving a continuous
piecewise-quadratic transition.  The construction is Bentkus-inspired; the
capped payoff is not a convex Bentkus test function.

The main figure contains the fixed Bentkus hedge, Bentkus-STaR,
the original and STaR product rules, target-capped quadratic STaR, and
Probit-STaR.  A separate matched-feedback ablation holds the chronological
clock, empirical-variance estimate, solvency cap, and target stopping fixed,
then changes only the feedback map among product square-root, squared-hinge,
and target-capped quadratic feedback; it also shows Probit
alongside the Gaussian limit.  This ablation isolates feedback behavior.

Confidence-set inversion uses endpoint searches, with a batched geometric
first-crossing scan for the target-capped feedback.  The run writes:

- the normalized- and raw-width comparisons to
  `plots/ci_width_original_vs_star.png` and
  `plots/ci_width_raw_original_vs_star.png`; and
- the normalized- and raw-width matched-feedback ablations to
  `plots/ci_width_feedback_ablation.png` and
  `plots/ci_width_raw_feedback_ablation.png`; and
- all numerical summaries to
  `plots/ci_width_original_vs_star.json`.

To redraw all four figures from that JSON without rerunning the simulations,
use

```bash
python plot_saved_experiment.py
```

The earlier exact-Bernoulli and Gaussian digital-DP benchmarks remain
available through `run_dp_experiment(...)`, which also includes Probit-STaR,
but they are not part of the default comparison.

The local-Gaussian comparison in the paper is reproducible with

```bash
python feedback_local_power.py
```

The script solves the common feedback boundary-value problem on nested
logit meshes, reports the product, capped-exponential, squared-hinge,
capped-quadratic, and digital slopes at one-sided level `0.005`, and isolates the feedback
crossing.
Its mesh-refinement and Richardson diagnostics are high-accuracy numerical
checks, not formal interval-arithmetic certificates.

Two implementation details support the comparisons in the paper:

- `compute_M_bets(...)` and `bets_ci_endpoints(...)` are matched fixed-plan
  product comparators. They use the same variance estimate, solvency cap, and
  target stopping as product STaR, differing only in whether the target and
  horizon are recalculated.
- `probit_star_randomized_ci_endpoints(...)` implements the Gaussian digital-
  delta feedback approximation. The experiment uses an RNG stream separate
  from the data stream, so adding the terminal randomizers
  does not change the other curves. The helper returns the accepted component
  containing the sample mean; the finite-sample coverage guarantee formally
  applies to the full inverted set. If randomization rejects the sample mean,
  the reported center component is empty (width zero), and its frequency is
  stored as `probit_empty_rate` rather than redrawing the uniforms.
- `capped_exponential_feedback_star_ci_endpoints(...)` caps and re-prices
  the exponential planning claim while retaining original STaR's
  state-dependent slope. `hinge_feedback_star_ci_endpoints(...)` and
  `capped_feedback_star_ci_endpoints(...)` provide the squared-hinge and
  capped-quadratic counterparts. All run inside the same one-clock product
  recursion and form the controlled feedback ablation.

The archived polynomial Constructions 1 and 2 and their clipping diagnostics
live in `legacy_constructions.py`. They are retained for reproducibility and are
not imported by the current experiment.

Run the script from this directory because it writes figures to `plots/`.
Selected publication figures can then be copied into `../paper/plots/`.

The experiment parameters are in `run_experiment(...)` and can be overridden
from Python for faster smoke tests.

## Confidence-sequence experiment

The horizon-free comparison is implemented separately in
`confidence_sequences.py`; it deliberately excludes STaR and dynamic
programming and uses the observed chronological order.  It compares five
procedures:

- finite \(G=20\) hedged grid-Kelly, retained as a historical diagnostic;
- a product scale mixture with
  \(r_j=2^{-j/2}\) and \(q_j=\{\zeta(2)j^2\}^{-1}\);
- product aGRAPA;
- a stopped Bentkus maturity mixture; and
- the same heat mixture with each directional stake capped by aGRAPA.

The product scales satisfy \(r_j^{-2}=2^j\), matching the geometric indices
of the Bentkus maturity grid; the product components' statistical horizons
also contain candidate-mean and variance constants.  Both countable
constructions are implemented through a declared maximum scale, with all
omitted prior mass held as cash.  The driver constructs
each finite schedule once and reuses it for every prefix; calling the product
scale convenience function afresh on growing prefixes would instead change
the planned truncation.

Run the publication driver with, for example,

```bash
python run_confidence_sequence_experiments.py \
  --max-time 100000 \
  --num-width-sims 30 \
  --coverage-max-time 10000 \
  --num-coverage-sims 5000 \
  --topology-grid-size 129 \
  --output-prefix confidence_sequences_t100k_final \
  --progress
```

The exploratory high-horizon extension reported in the paper is reproduced
with

```bash
python run_confidence_sequence_experiments.py \
  --delta 0.01 \
  --max-time 1000000 \
  --times 1000,10000,100000,1000000 \
  --num-width-sims 5 \
  --coverage-max-time 10000 \
  --num-coverage-sims 0 \
  --topology-grid-size 129 \
  --product-grid-size 20 \
  --seed 20260718 \
  --output-prefix confidence_sequences_t1m_final \
  --progress
```

This run is a five-path large-scale trend check; the separate 5,000-path
experiment above supplies the coverage assessment.

The driver writes complete JSON results plus absolute-width and
relative-width figures, and an anytime-error figure when coverage simulation
is requested, under `plots/`.  Every reported
running intersection uses the e-process maximum over all intervening
observations, not just the plotted checkpoints.  The Bentkus schedule uses
geometric maturities, polynomial prior weights, and retains the omitted tail
mass as cash, so finite truncation remains a unit-initialized e-process.
Endpoint inversion is numerical: the JSON therefore retains per-path empty,
disconnection, and topology-uncertainty diagnostics, and the paper separately
checks stability across scan grids.

Reproduce the 33/65/129 topology-resolution audit and the Bentkus maturity
prior sensitivity study by running:

    python run_confidence_sequence_audits.py --audit all --progress

This writes strict-JSON audit records under plots/; all compared grids,
schedules, and methods reuse the same chronological paths within an audit.
The separate fixed-strike diagnostic used in the paper is reproduced by:

    python run_bentkus_strike_audit.py

It compares the theoretically calibrated strike with fixed multipliers and
less stringent effective-level choices, then evaluates the best fixed
multiplier against product betting on the same small set of paths.

Run the dedicated tests with

```bash
python -m unittest -v test_confidence_sequences.py
```
