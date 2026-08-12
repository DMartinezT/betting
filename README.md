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

The driver evaluates the following fixed-horizon procedures on nine bounded
distributions:

- the original two-sided product test martingale;
- the target-recalculating STaR betting rule from
  [STaR-bets-confidence-interval](https://github.com/vvoracek/STaR-bets-confidence-interval),
  used inside the same two-sided product test;
- Efficient betting, the Gaussian-digital feedback rule proposed in the
  paper, using the unbuffered recursion and one fixed terminal uniform per arm during
  each confidence-set inversion;
- common-clock Efficient betting, which uses the same feedback and terminal
  uniforms but shares one predictable residual-variance estimate across both
  arms and all candidate means, giving a pathwise interval inversion;
- Construction 3, the original nonnegative Bentkus/heat-flow hedge;
- a target-recalculating Bentkus hedge that re-optimizes the local
  squared-hinge strike from current wealth, remaining variance, and the fixed
  rejection target;
- a matched-clock squared-hinge feedback rule; and
- a matched-clock target-capped quadratic feedback rule used in the appendix.

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

The main fixed-sample figure contains the fixed Bentkus hedge, Bentkus-STaR,
the original and STaR product rules, and common-clock Efficient betting.  It
plots confidence-interval widths: a mesh-inverted raw set is replaced by its
convex hull, while the common-clock inversion is interval-valued pathwise.
Candidate-dependent Efficient betting and the matched-feedback ablation are
kept for appendix comparisons.

`betting.py` first writes fast adjacent-component endpoints.  To obtain the
publication width convention, replay the same data and terminal-randomizer
streams with the topology-aware postprocessor, then redraw:

```bash
python augment_fixed_sample_topology.py
python plot_saved_experiment.py
```

The postprocessor does not assume convexity for the candidate-dependent
procedures.  Its adaptive finite-mesh
inversion combines a standard-error-scale grid, global and geometric probes,
and exponential refinement after any detected fragmentation.  It records the
full-set diameter, the largest accepted-component width, component counts,
mesh resolution, and point-budget diagnostics.  Figures 2 and 4 use 120
topology replications per distribution through `n=1000`, 60 through
`n=10000`, and 30 thereafter.  The plots show mean full-set diameters,
equivalently convex-hull CI widths; percentile and largest-component
summaries remain in the JSON.  A finite mesh can still miss two crossings
inside one final cell, so the saved diagnostics are part of the result rather
than a proof of connectedness.  The common-clock procedure is inverted
directly from its two monotone arm boundaries, so its diameter and
largest-component width coincide exactly and require no discovery mesh.  The
run writes:

- the normalized- and raw-width comparisons to
  `plots/ci_width_original_vs_star.png` and
  `plots/ci_width_raw_original_vs_star.png`; and
- the appendix-only Bentkus-STaR comparison to
  `plots/ci_width_bentkus_star_vs_original_star.png`; and
- the normalized- and raw-width matched-feedback ablations to
  `plots/ci_width_feedback_ablation.png` and
  `plots/ci_width_raw_feedback_ablation.png`; and
- all numerical summaries to
  `plots/ci_width_original_vs_star.json`.

To redraw the figures from the augmented JSON without rerunning either the
simulations or the topology inversion, use `python plot_saved_experiment.py`.
The deterministic three-panel introductory comparison and the full
betting--Gaffke comparison use the dedicated Gaffke sample and are generated
together by

```bash
python gaffke_comparison/figure2_gaffke.py
python plot_combined_main_figure.py
```

The plotting command writes `plots/intro_deterministic_comparison.png` and
`plots/ci_width_original_vs_star.png`, copying both into the paper repository.


The Appendix D finite-sample comparison adds Poisson-efficient (PE) betting,
the skew-corrected GE rule of Definition D9.  Its predictable clock matches
the variance and third cumulant to a compensated Poisson device, uses
real-order regularized-incomplete-gamma tails, and falls back exactly to GE
when the estimated squared skewness is below the configured threshold.  The
implementation deliberately enforces `n <= 10000`.  Reproduce the paired
120-path comparison through `n=10000` and copy its figure into the paper with

```bash
python poisson_betting_figure.py
```

The default run replays the Figure 3 data and terminal-randomizer streams,
uses `c=0.5`, third-moment shrinkage `t0=10`, and fallback threshold
`epsilon=0.05`, and writes the interval-level audit, summary, configuration,
and figure under `plots/poisson_betting/`.  It also copies
`ci_width_poisson_betting.png` to `../paper/plots/`.  Use `--resume` to
continue from `pe_intervals.csv` or `--plot-only` to redraw without
rerunning the simulation.

Appendix D also includes matched sensitivity plots at `c=0.75` and `c=1`.
Their PE audits live under `plots/poisson_betting_c075/` and
`plots/poisson_betting_c1/`; the paper-facing figures are named
`ci_width_poisson_betting_c05.png`,
`ci_width_poisson_betting_c075.png`, and
`ci_width_poisson_betting_c1.png`.  The plotting script accepts
`--solvency-c`, `--input`, `--output-dir`, and `--plot-name` so each variant
is checked against betting summaries computed with the same common solvency
fraction.

The large-sample main plots, including 30 paired common-clock intervals per
distribution and horizon, are produced by

```bash
python gaffke_comparison/large_sample_feedback_gaffke.py --resume
python gaffke_comparison/augment_large_sample_topology.py
python gaffke_comparison/large_sample_feedback_gaffke.py --plot-only
```

This audit uses 30 paired paths per distribution and horizon for every main
method.  Through `n=10000` it uses the full multiresolution scan.  At larger
horizons it reuses the accurately bisected local component, checks 31 evenly
spaced interior points for gaps, and probes outside at standard-error distances
`1/8, 1/4, 1/2, 1, 2, ...`, together with 0 and 1.  Any interior rejection or
outside acceptance triggers the full scan.  This hybrid screen keeps the work
manageable through `n=10**7`.  The main Gaffke figure shows convex CI widths
and uses only common-clock Efficient betting.  The focused appendix figure
adds candidate-dependent Efficient betting, with full-set diameters and
largest-component widths retained separately in the saved results.

Finite-sample confidence sets are not always intervals.  Run the global
topology audit with

```bash
python audit_confidence_set_topology.py
```

The audit evaluates the complete candidate-mean range on successively refined
meshes, refines every visible crossing, and writes
`plots/confidence_set_topology_audit.json`.  It reports the component list,
the sum of component lengths, the full-set diameter (equivalently the width
of its convex hull), the largest-component length, and the center-component
length.  The convex hull remains
a valid confidence interval because it contains the original confidence set.
As with any finite grid, the audit can miss multiple crossings contained
inside one mesh cell, so topology claims should be checked across resolutions.
The underlying helpers are `_confidence_set_components(...)`,
`_confidence_set_widths(...)`, and `_confidence_set_hull_endpoints(...)`.

The earlier exact-Bernoulli and Gaussian digital-DP benchmarks remain
available through `run_dp_experiment(...)`, which also includes Efficient betting,
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
- `compute_M_heat_star_arms(...)` target-caps the two common-clock
  squared-hinge STaR arms, and `heat_star_common_clock_ci_endpoints(...)`
  inverts their ordered rejection boundaries directly.  This is the code path
  covered by the squared-hinge concavity lemma.
- `probit_star_randomized_ci_endpoints(...)` implements the Gaussian digital-
  delta feedback approximation. The experiment uses an RNG stream separate
  from the data stream, so adding the terminal randomizers
  does not change the other curves. The endpoint helper returns the accepted
  component containing the sample mean, whereas
  `_confidence_set_components(...)` performs the global inversion.  If
  randomization rejects the sample mean, the adjacent component is empty
  (width zero), and its frequency is stored as `probit_empty_rate` rather than
  redrawing the uniforms.
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

## Fixed-event survival confidence intervals

`survival_fixed_event.py` compares terminal 95% log-hazard-ratio intervals
after exactly 200 failures.  It does not monitor methods before the terminal
event count and does not use external terminal randomization.  The design
crosses four hazard patterns (null, proportional HR 0.70, delayed effect, and
crossing hazards) with no, balanced, and arm-differential independent
censoring.

The five methods are the classical logrank score interval, GE-logrank, the
published AV prequential plug-in interval, the published eventwise two-sided
conditional-GROW process, and a once-at-time-zero mixture of the two complete
directional AV point-alternative processes.  The last two are intentionally
kept separate: averaging the directional likelihood ratios at every event is
not the same as averaging their complete products.

Reproduce the paper experiment (10,000 replications in each of 12 scenarios)
with

```bash
python survival_fixed_event.py --repetitions 10000 --events 200
```

The command writes `summary.json`, compressed replicate-level endpoints, and
the publication figure under `plots/survival_fixed_event/`, and copies the
figure to `../paper/plots/`.  On the development machine (AMD Ryzen 9 9950X),
the audited run took about 65 seconds end to end.  Run the dedicated tests
with

```bash
python -m unittest -v test_survival_fixed_event.py
```
