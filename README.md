# Online Mixture of Autoregressive Experts

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

- `data/synthetic.py`: Generates seeded AR(4) observations and returns the real generating coefficients.
- `experts/scalar.py`: Calculates scalar AR coefficients using Levinson–Durbin and checks them with a direct solve.
- `experts/online.py`: Stores and updates the eight predictors' coefficients and recent observations.
- `experiments/stationary.py`: Runs the stationary experiment and saves figures and numerical results.
- `experiments/gate_example.py`: Shows where Tasnuba's gate connects; run with `python -m experiments.gate_example`.
- `tests/test_pipeline.py`: Tests coefficient calculations, reproducibility, prediction timing, and failed fits. Run with `python -m unittest discover -s tests -v`.
- `data/generated/`: Contains saved synthetic observations and real generating coefficients.
- `outputs/`: Contains forecasts, figures, and results, grouped by experiment, channel count, and seed.

Only the stationary scalar experiment is implemented. Real-data loaders, multichannel experts, switching/drifting runners, `gating/`, and `evaluation/` will be added as that work is implemented.
