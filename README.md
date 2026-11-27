# Online Mixture of Autoregressive Experts

Run these commands from the repository root with the packages in `requirements.txt` installed. All figures are saved as **PDF only**. Repeating a command replaces the same files.

## Synthetic data

### Reproduce the data figures

```powershell
python -m experiments.generate_synthetic
```

Generates 21 datasets: stationary AR(4) and switching data with `T=500` and `T=2000` for `n=1` and `n=3`, plus scalar drifting AR(1), using seeds 685, 686 and 687. Each keeps 10,000 observations; every stationary/switching regime discards its own 200 burn-in values. The drifting series uses 200 initial burn-in steps at coefficient 0.9, then follows 0.9 to -0.9 to 0.8 without restarting at the jump.

Observations and real coefficients are saved separately under `data/generated/`, grouped by dataset, channel count, switching interval and seed. Seven `data_overview.pdf` figures use seed 685 under `outputs/synthetic_data/`. `generation_checks.json` records the checks for all 21 datasets. These figures inspect the generated data; they do not show forecasting performance.

## Stationary AR(4) experts

### Reproduce the expert figures

```powershell
python -m experiments.stationary
```

Fits scalar AR(1)..AR(8) on stationary AR(4) observations. Four PDFs are saved in each `outputs/stationary/n1/seed_685/`, `seed_686/` and `seed_687/` folder:

- `coefficient_paths.pdf`: Estimated coefficients over time against the real coefficients.
- `variance_by_order.pdf`: Estimated error variance and forecast MSE by expert order.
- `data_diagnostics.pdf`: Observation trace and autocorrelation.
- `alignment_and_guards.pdf`: Forecasts against targets and failed-fit activity.

### Results

AR(4) had the lowest average forecast MSE across the three seeds: **1.0345**. Higher orders reduced fitted error variance but increased average forecast MSE, so better fit did not necessarily produce better predictions.

## Levinson-Durbin test against a direct solve

```powershell
python -m unittest discover -s tests -v -k LevinsonTests
```

Checks the recursion against `np.linalg.solve` using identical covariance estimates and regularization, including white-noise, AR(1) and AR(4) cases. This is the direct-solve check required by Section 1.4.

## Files

- `data/synthetic.py`: Generates observations and returns the real generating coefficients separately.
- `experts/scalar.py`: Fits scalar AR coefficients and provides the direct-solve check.
- `experts/online.py`: Stores and updates AR(1)..AR(8) using past observations.
- `experiments/generate_synthetic.py`: Saves synthetic datasets and their data figures.
- `experiments/stationary.py`: Saves stationary expert forecasts, figures and results.
- `experiments/gate_example.py`: Shows where Tasnuba's gate connects to the experts.
- `tests/test_pipeline.py`: Checks fitting, forecast timing and failed updates.
- `tests/test_synthetic.py`: Checks seeds, channel coupling, burn-in and coefficient changes.

## Remaining report figures

Sections 1.6-1.7 also require stationary error-variance behavior over time, gate weights against switching orders, forecast-error curves, regret tables and ablation results. Those remaining results, multichannel experts and real-data experiments are not yet complete. The data-overview and debugging figures above are additional sanity checks.
