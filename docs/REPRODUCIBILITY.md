# Manuscript reproducibility

This repository separates two tasks:

1. **Figure regeneration** redraws every manuscript figure from committed
   summaries and replication-level results.  It is deterministic apart from
   harmless backend metadata and does not run simulations.
2. **Experiment regeneration** reruns Monte Carlo experiments from their
   recorded seeds and settings.  Some publication runs take minutes or hours.

## Redraw every manuscript figure

From the repository root, run:

```bash
python reproduce_paper_figures.py
```

The command validates all cached inputs, renders into a temporary directory,
and copies the eleven successful outputs to `paper_figures/`.  To write
directly into a neighboring checkout of the paper, use:

```bash
python reproduce_paper_figures.py --output-dir ../paper/plots
```

Useful inspection commands are:

```bash
python reproduce_paper_figures.py --list
python reproduce_paper_figures.py --check-inputs
python reproduce_paper_figures.py --only intro_both_comparison.png
```

## Figure manifest

| Output file | Source module | Committed plotting input |
| --- | --- | --- |
| `intro_both_comparison.png` | `figures/plot_combined_main_figure.py` | fixed-sample JSON and dedicated Gaffke results |
| `wor_fixed_fraction_intro_comparison.png` | `experiments/wor_scaled_plots.py` | `plots/wor/fixed_fraction_sqrt_n_widths.csv` |
| `efficient_betting_function.pdf` | `figures/plot_efficient_betting_function.py` | analytic; no saved data |
| `ci_width_original_vs_star.png` | `figures/plot_combined_main_figure.py` | fixed-sample JSON and dedicated Gaffke results |
| `wor_fixed_horizon_widths.png` | `wor.py` | `plots/wor/fixed_horizon_widths.csv` |
| `wor_fixed_fraction_sqrt_n_widths.png` | `experiments/wor_scaled_plots.py` | `plots/wor/fixed_fraction_sqrt_n_widths.csv` |
| `scaled_width_star_efficient_shared_estimator_comparison.png` | `gaffke_comparison/large_sample_feedback_gaffke.py` | c=1 large-sample results and config |
| `ci_width_solvency_combined.png` | `figures/plot_solvency_sensitivity.py` | c=1/2, 3/4, and 1 fixed-sample JSONs |
| `ci_width_fixed_hinge_vs_product.png` | `figures/plot_saved_experiment.py` | unified c=1 fixed-sample JSON |
| `horizon_free_wr_cs_widths.png` | `experiments/horizon_free_wr_cs.py` | planned-window summary CSV |
| `horizon_free_cs_widths.png` | `experiments/horizon_free_cs.py` | finite-population summary CSV |

The Python manifest in `reproduce_paper_figures.py` is authoritative and is
covered by unit tests.

## Rerun the underlying experiments

The primary drivers all record their seed and settings in JSON configuration
or result files:

```bash
# Fixed-horizon iid experiment and topology-aware inversion
python betting.py
python -m experiments.augment_fixed_sample_topology

# Dedicated Gaffke samples and large-horizon comparison
python gaffke_comparison/figure2_gaffke.py
python gaffke_comparison/large_sample_feedback_gaffke.py --resume
python gaffke_comparison/augment_large_sample_topology.py

# Sampling without replacement
python wor.py --run-experiments
python -m experiments.wor_scaled_plots --run-experiments

# Confidence sequences
python -m experiments.horizon_free_wr_cs --run-experiments
python -m experiments.horizon_free_cs --run-experiments
```

The saved fixed-sample c-sensitivity files were produced with matched seeds,
horizons, replication counts, and terminal randomizers.  Their metadata lists
the recomputed methods and solvency fractions.  Regenerating those expensive
variants should use separate `--output` and `--checkpoint` paths with
`experiments/augment_fixed_sample_topology.py`; never overwrite a publication result
until the run and metadata have been audited.

## Randomness and numerical conventions

- Data streams and external terminal randomizers use separate seeded NumPy
  generators.
- A terminal randomizer pair is fixed over all candidate means in one
  confidence-set inversion.
- Saved configurations include simulation counts, horizon grids, calibration,
  and inversion conventions.
- Historical result files retain their original serialized method keys even
  where current figures display the unified name `GE-betting`.
- Figure regeneration should be run with the pinned versions in
  `requirements.txt` when byte-level stability matters.
