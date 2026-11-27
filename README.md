# Online Mixture of Autoregressive Experts

## Synthetic data

From the repository root, run:

```powershell
python -m experiments.generate_synthetic
```

This saves 21 datasets: stationary AR(4) and switching AR data with `T=500` and `T=2000` for both `n=1` and `n=3`, plus the scalar drifting AR(1), for seeds 685, 686, and 687. Every dataset keeps 10,000 observations with independent unit-variance Gaussian noise. Switching orders cycle through 2, 6, 3, and 8; each regime discards its own 200 burn-in values.

Observations and real coefficients are separate files under `data/generated/stationary/n1/seed_685/`, `data/generated/switching/n3/T500/seed_685/`, and equivalent folders for the other settings. Drifting data is under `data/generated/drifting/n1/seed_685/`.

Data figures use seed 685 and are saved under `outputs/synthetic_data/`. `generation_checks.json` records shape, stability and seeded-noise checks for every dataset. These are data checks, not forecasting results. The drifting AR(1) uses 200 initial burn-in steps at coefficient 0.9, then drifts to -0.9 and jumps to 0.8 without restarting.

- `data/synthetic.py`: Generates the stationary, switching and drifting observations and returns the real coefficients separately.
- `experiments/generate_synthetic.py`: Saves all datasets and data-check figures.
- `tests/test_synthetic.py`: Checks channel coupling, regime boundaries, burn-in, seeds, noise and the drift/jump path.

## Stationary AR(4)

Stationary, single-channel AR(4) data with online AR(1) through AR(8) predictors.

### Reproduce the figures

From the repository root, run:

```powershell
python -m experiments.stationary
```

Figures are saved as PNG and PDF files under `outputs/stationary/n1/seed_685/`, with equivalent folders for seeds 686 and 687. The observations and real generating coefficients are saved separately under `data/generated/stationary/n1/seed_685/`, again with one folder per seed.

### Results

AR(4) had the lowest average forecast MSE across the three seeds: **1.0345**. Higher orders reduced the fitted error variance but increased average forecast MSE, showing that a better fit did not necessarily produce better predictions.

### Files

- `experts/scalar.py`: Calculates scalar AR coefficients using Levinson–Durbin and checks them with a direct solve.
- `experts/online.py`: Stores and updates the eight predictors' coefficients and recent observations.
- `experiments/stationary.py`: Runs the stationary experiment and saves figures and numerical results.
- `experiments/gate_example.py`: Shows where Tasnuba's gate connects; run with `python -m experiments.gate_example`.
- `tests/test_pipeline.py`: Tests coefficient calculations, reproducibility, prediction timing, and failed fits. Run with `python -m unittest discover -s tests -v`.
- `data/generated/`: Contains saved synthetic observations and real generating coefficients.
- `outputs/`: Contains forecasts, figures, and results, grouped by experiment, channel count, and seed.

Synthetic data generation and the stationary scalar expert experiment are implemented. Multichannel experts, forecasting experiments on switching/drifting data, real-data loaders, `gating/`, and `evaluation/` remain future work.
