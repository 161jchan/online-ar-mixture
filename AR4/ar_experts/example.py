"""Show Tasnuba how to run the generator and experts together.

Generate the stationary AR(4) data, initialize AR(1)..AR(8), then repeatedly
predict, score the forecasts, and update the experts.

This file is a runnable usage example. Tasnuba can follow the same calls
in her own runner and insert her gate at the marked points.
"""

import numpy as np
from .data import DataConfig, generate_stationary_ar4
from .experts import OnlineARExperts


def stream_with_gate(values, gate_predict=None, gate_update=None):
    """gate_predict and gate_update are functions Tasnuba can supply.
    Without them, this example runs only the experts and records their errors.
    """
    bank = OnlineARExperts().initialize(values[:40])
    errors = []
    for target_raw in values[40:]:
        # Get AR(1)..AR(8)'s forecasts before using the next actual observation.
        forecasts_z = bank.predict()
        before = bank.snapshot()
        # Let Tasnuba's optional gate combine the eight forecasts.
        mixture_z = None if gate_predict is None else gate_predict(forecasts_z.copy(), before)
        target_z = bank.standardize(target_raw)
        # Compare the actual observation with the saved forecasts before
        # updating either AR(1)..AR(8) or the gate.
        errors.append((target_z-forecasts_z[:, 0])**2)
        # Update the experts for the following observation.
        bank.update(target_raw)
        # Update the optional gate using the forecasts saved before the update.
        if gate_update is not None:
            gate_update(target_z.copy(), forecasts_z.copy(), before, mixture_z)
    return np.asarray(errors), bank.snapshot()


def main():
    values, _truth = generate_stationary_ar4(DataConfig())
    errors, final = stream_with_gate(values)
    print('Retained observations:', len(values), '| scored:', len(errors))
    print('Per-expert MSE (standardized):', np.mean(errors, axis=0))
    print('Final normalization:', final['normalization'])
    print('Connect gate_predict and gate_update hooks in stream_with_gate().')


if __name__ == '__main__':
    main()
