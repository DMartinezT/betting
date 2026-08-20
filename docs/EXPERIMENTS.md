# Experiment catalog

The root-level Python files are retained as stable research entry points.
They are grouped here by purpose so exploratory and publication workflows do
not get mixed together.

## Core implementations

| Module | Purpose |
| --- | --- |
| `betting.py` | Fixed-horizon product, STaR, squared-hinge, Poisson-efficient, and GE-betting wealth processes and inversions. |
| `confidence_sequences.py` | With-replacement confidence-sequence processes and numerical inversion. |
| `wor.py` | Sampling-without-replacement bridge clock, competitors, and fixed-horizon experiment. |
| `robust_studentized_dp.py` | Bellman recursion used by the robust Studentized-event audit. |
| `legacy_constructions.py` | Archived polynomial constructions retained for historical reproduction. |

## Manuscript experiment drivers

| Driver | Output location | Role |
| --- | --- | --- |
| `plot_combined_main_figure.py` | `plots/` | Introductory and nine-distribution iid figures. |
| `plot_efficient_betting_function.py` | `plots/` | Analytic GE-betting fraction figure. |
| `plot_saved_experiment.py` | selected fixed-sample result directory | Fixed squared-hinge and feedback-ablation figures. |
| `plot_solvency_sensitivity.py` | `plots/solvency_c_comparison/` | Solvency-fraction comparison. |
| `wor_scaled_plots.py` | `plots/wor/` | Fixed-fraction and sqrt(n)-scaled finite-population figures. |
| `horizon_free_wr_cs.py` | `plots/horizon_free_wr_cs/` | Planned-window confidence sequences with replacement. |
| `horizon_free_cs.py` | `plots/horizon_free_cs/` | Confidence sequences without replacement. |
| `gaffke_comparison/` | its `*_results/` directories | Gaffke simulations, large-sample comparisons, and topology postprocessing. |

Use `reproduce_paper_figures.py` for manuscript figures.  Calling individual
drivers is mainly useful when rerunning or extending one experiment.

## Auxiliary and diagnostic experiments

| Driver | What it investigates |
| --- | --- |
| `audit_confidence_set_topology.py` | Connected components and convex-hull widths on refined candidate grids. |
| `compare_markov_calibrations_large.py` | Deterministic versus randomized Markov calibration. |
| `figure3_coverage.py` | Finite-sample noncoverage diagnostics. |
| `feedback_local_power.py` | Numerical local-Gaussian feedback comparison. |
| `poisson_betting_figure.py` | Skew-corrected Poisson-efficient betting. |
| `order_invariant_ge.py` | Permutation-integrated iid GE construction and order sensitivity. |
| `robust_studentized_experiment.py` | Robust symmetric Studentized-event benchmarks. |
| `run_confidence_sequence_experiments.py` | Earlier confidence-sequence width and coverage experiments. |
| `run_confidence_sequence_audits.py` | Confidence-sequence topology and schedule audits. |
| `run_bentkus_strike_audit.py` | Fixed-strike and effective-level sensitivity. |
| `survival_fixed_event.py` | Fixed-event survival confidence intervals. |

Every driver supports either explicit command-line options, a `--help`
message, or a documented Python entry point.  Final outputs live under
`plots/`; resumable checkpoint files are intentionally ignored by Git.

## Fixed-horizon iid workflow

The publication-scale fixed-horizon experiment uses

```bash
python betting.py
python augment_fixed_sample_topology.py
python plot_saved_experiment.py
```

`betting.py` evaluates the common sample-size grid
`10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000`
on the nine manuscript distributions.  Its default replication counts are
stored in `PUBLICATION_SIMULATION_COUNTS`, and the output JSON records the
resolved counts, horizons, seed, inversion settings, and method settings.

The topology postprocessor replays the recorded data and terminal-randomizer
streams.  For candidate-dependent procedures it records the full-set
diameter, largest accepted component, component count, mesh resolution, and
point budget; common-clock GE-betting is inverted directly from its two
ordered arm boundaries.  The plotting step can then redraw the saved results
without rerunning either computation.

The dedicated Gaffke sample and the combined introductory/main figure are
generated with

```bash
python gaffke_comparison/figure2_gaffke.py
python plot_combined_main_figure.py
```

For the large-horizon estimator comparison, use

```bash
python gaffke_comparison/large_sample_feedback_gaffke.py --resume
python gaffke_comparison/augment_large_sample_topology.py
python gaffke_comparison/large_sample_feedback_gaffke.py --plot-only
```

These runs are expensive.  A checkpoint is written locally for safe resume,
but only the final configuration and result table are versioned.

## Sampling without replacement and confidence sequences

The current finite-population fixed-horizon and fixed-fraction experiments
are run with

```bash
python wor.py --run-experiments
python wor_scaled_plots.py --run-experiments
```

The planned-window with-replacement and horizon-free finite-population
confidence-sequence experiments are run with

```bash
python horizon_free_wr_cs.py --run-experiments
python horizon_free_cs.py --run-experiments
```

Each of these drivers supports `--plot-only` to redraw its figure from the
committed summary.  The authoritative manuscript command calls the same
plotting functions with isolated output paths.

## Auxiliary reproduction recipes

### Poisson-efficient comparison

The skew-corrected Poisson-efficient comparison through `n=10000` is:

```bash
python poisson_betting_figure.py
```

It writes interval-level audits, summaries, configuration, and figures under
`plots/poisson_betting/`.  Use `--resume` for a partial simulation or
`--plot-only` to redraw it.  The driver also accepts `--solvency-c`,
`--input`, `--output-dir`, and `--plot-name`; the saved `c=0.75` and
`c=1` variants live in the corresponding `plots/poisson_betting_*/`
directories.

### Confidence-set topology

```bash
python audit_confidence_set_topology.py
```

This global audit refines visible crossings over the full candidate-mean
range and records accepted components, total accepted length, convex-hull
diameter, largest-component length, and center-component length.  Because it
uses a finite mesh, the output is a numerical diagnostic rather than a proof
of connectedness.

### Earlier confidence-sequence comparison

The earlier long-horizon comparison in `confidence_sequences.py` is driven
by:

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

The exploratory million-step trend check used:

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

Run `python run_confidence_sequence_audits.py --audit all --progress` for
the topology-resolution and schedule audits, and
`python run_bentkus_strike_audit.py` for the fixed-strike sensitivity
diagnostic.

### Local feedback and fixed-event survival

`python feedback_local_power.py` runs the nested-logit-mesh comparison of
the local feedback slopes.  The fixed-event survival experiment is:

```bash
python survival_fixed_event.py --repetitions 10000 --events 200
```

It writes the summary, compressed replicate-level endpoints, and figure under
`plots/survival_fixed_event/`.

### Order invariance

```bash
python order_invariant_ge.py \
  --output-dir plots/order_invariant_ge \
  --n-values 50 200 1000 5000 \
  --repetitions 100 \
  --permutations 32 \
  --seed 20260813
```

This iid-only diagnostic writes pathwise, paired, order-sensitivity, and
monotonicity audits plus two figures.  Its finite permutation sample is part
of the external randomization.

### Robust Studentized-event audit

```bash
python robust_studentized_experiment.py \
  --output-dir plots/robust_studentized \
  --n-values 50 200 1000 5000 \
  --repetitions 200 \
  --seed 20260814 \
  --bisection-steps 15 \
  --paper-plot ../paper/plots/robust_studentized_widths.png
```

This workflow combines the exact finite-grid Bellman calibration with
pathwise width and paired-summary outputs.  The implementation itself is in
`robust_studentized_dp.py`.
