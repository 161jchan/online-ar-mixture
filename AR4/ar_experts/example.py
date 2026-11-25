"""Run actual required stationary data; replace gate hooks with Tasnuba's gate."""

import numpy as np
from .data import DataConfig, generate_stationary_ar4
from .experts import OnlineARExperts


def stream_with_gate(values, gate_predict=None, gate_update=None):
    """Hooks are optional; no gate/backbone is implemented in this milestone.

    gate_predict(forecasts_z, snapshot) returns a scalar mixture forecast z.
    gate_update(target_z, saved_forecasts_z, saved_snapshot, mixture_z) handles
    the gate's post-observation update using the saved pre-observation inputs.
    With no hooks, simply record the eight experts' forecast errors.
    """
    bank = OnlineARExperts().initialize(values[:40])
    errors = []
    for target_raw in values[40:]:
        forecasts_z = bank.predict()  # (8,1), using only earlier observations
        before = bank.snapshot()
        mixture_z = None if gate_predict is None else gate_predict(forecasts_z.copy(), before)
        # The target is revealed here. It was not passed to gate_predict.
        target_z = bank.standardize(target_raw)
        errors.append((target_z-forecasts_z[:, 0])**2)
        # Record errors BEFORE changing either expert or gate state.
        bank.update(target_raw)
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
