# Online Mixture of Autoregressive Experts

## AR(4)

Stationary AR(4) data with online AR(1) through AR(8) predictors.

### Reproduce the figures

From the repository root, run:

```powershell
cd AR4
python -m ar_experts.diagnostics
```

Figures are saved as PNG and PDF files in `AR4/outputs/seed_685/`, `seed_686/`, and `seed_687/`.

### Results

AR(4) had the lowest average forecast MSE across the three seeds: **1.0345**. Higher orders reduced the fitted error variance but increased average forecast MSE, showing that a better fit did not necessarily produce better predictions.

### Files

- `ar_experts/data.py`: Generates seeded AR(4) observations and saves the real generating coefficients.
- `ar_experts/levinson.py`: Calculates AR coefficients using Levinson–Durbin and checks them with a direct solve.
- `ar_experts/experts.py`: Stores and updates the eight predictors’ coefficients and recent observations.
- `ar_experts/example.py`: Shows where Tasnuba’s gate connects to the prediction-and-update loop.
- `ar_experts/diagnostics.py`: Runs the stationary experiment and saves figures and numerical results.
- `ar_experts/__init__.py`: Makes the generator and expert classes available for import.
- `tests/test_pipeline.py`: Tests coefficient calculations, reproducibility, prediction timing, and failed fits.
- `outputs/`: Contains generated observations, forecasts, figures, and results for each seed.
