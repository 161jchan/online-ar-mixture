"""Generate and check all required synthetic datasets, without fitting experts."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
from data import DataConfig, generate_stationary_ar4, generate_switching_ar, generate_drifting_ar1
from data.synthetic import spectral_radius


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def check_series(values, truth):
    """Check lengths and the AR equation against the saved seed's Gaussian noise.

    Reconstruct the noise independently from its recorded random-stream key.
    For switching data, use each regime's own starting lags at the boundary.
    A concatenated boundary does not use the previous regime's observations.
    """
    config = truth['configuration']
    N, n = config['n_observations'], config['channels']
    if values.shape != (N,n) or not np.isfinite(values).all():
        raise AssertionError('Wrong observation shape or nonfinite observations')
    if not np.array_equal(np.asarray(truth['noise_covariance_raw']), np.eye(n)):
        raise AssertionError('Declared innovation covariance is not identity')
    recovered = np.empty_like(values)
    expected = np.empty_like(values)
    max_radius = 0.0
    if truth['kind'] == 'drifting':
        a = np.asarray(truth['coefficients'])
        required = np.r_[np.linspace(.9,-.9,N//2), np.full(N//2,.8)]
        np.testing.assert_array_equal(a, required)
        if truth['jump_index'] != N//2:
            raise AssertionError('Wrong AR(1) jump index')
        previous = np.r_[truth['retained_initial_lag'], values[:-1,0]]
        recovered[:,0] = values[:,0] - a*previous
        noise = np.random.default_rng(np.random.SeedSequence(
            config['seed'], spawn_key=tuple(truth['noise_spawn_key']))).normal(size=N+config['burn_in'])
        expected[:,0] = noise[config['burn_in']:]
        max_radius = float(np.abs(a).max())
        orders, switch_count = [1], 1
    else:
        if truth['kind'] == 'stationary':
            regimes = [{'start_index':0, 'stop_index':N, **truth}]
        else:
            T = truth['regime_length']
            regimes = truth['regimes']
            if truth['switch_indices'] != list(range(T,N,T)) or len(regimes) != N//T:
                raise AssertionError('Incorrect switching boundaries')
            if [r['order'] for r in regimes] != [(2,6,3,8)[i%4] for i in range(N//T)]:
                raise AssertionError('Incorrect generating-order cycle')
        cursor = 0
        for regime in regimes:
            start, stop = regime['start_index'], regime['stop_index']
            if start != cursor or stop <= start:
                raise AssertionError('Regimes overlap or leave a gap')
            a = np.asarray(regime['coefficients'])
            radius = spectral_radius(a)
            if radius > config['radius_cap']:
                raise AssertionError('Generating regime exceeds the stability cap')
            max_radius = max(max_radius, radius)
            history = np.array(regime['retained_initial_lags']).reshape(regime['order'],n)
            matrix = None if n == 1 else np.concatenate(a, axis=1)
            for t in range(start,stop):
                prediction = a @ history[:,0] if n == 1 else matrix @ history.reshape(-1)
                recovered[t] = values[t] - prediction
                history[1:] = history[:-1].copy()
                history[0] = values[t]
            noise = np.random.default_rng(np.random.SeedSequence(
                config['seed'], spawn_key=tuple(regime['noise_spawn_key']))).normal(
                    size=(stop-start+config['burn_in'],n))
            expected[start:stop] = noise[config['burn_in']:]
            if regime['simulated_observations'] != stop-start+config['burn_in']:
                raise AssertionError('Burn-in count mismatch')
            cursor = stop
        if cursor != N:
            raise AssertionError('Regimes do not cover the retained series')
        orders = [r['order'] for r in regimes]
        switch_count = len(regimes)-1
    equation_error = float(np.abs(recovered-expected).max())
    if equation_error > 1e-10:
        raise AssertionError('Observations do not match their coefficients and seeded noise')
    centered_noise = recovered-recovered.mean(axis=0)
    return {
        'kind': truth['kind'], 'channels': n, 'seed': config['seed'],
        'retained_observations': N, 'all_observations_finite': True,
        'regime_length': truth.get('regime_length'), 'orders': orders,
        'switch_count': switch_count, 'maximum_true_spectral_radius': max_radius,
        'burn_in_per_regime': config['burn_in'],
        'simulated_observations': N+config['burn_in']*(len(orders) if truth['kind']!='drifting' else 1),
        'innovation_equation_max_abs_error': equation_error,
        'innovation_mean': recovered.mean(axis=0).tolist(),
        'innovation_covariance_empirical': (centered_noise.T@centered_noise/N).tolist(),
        'raw_observation_mean': values.mean(axis=0).tolist(),
        'raw_observation_std': values.std(axis=0).tolist(),
        'observations_sha256': hashlib.sha256(values.tobytes(order='C')).hexdigest(),
    }


def plot_data(directory, values, truth, seed):
    """Plot the raw observations and the generating order or drifting coefficient."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    n, N = values.shape[1], len(values)
    time = np.arange(N)
    fig, axes = plt.subplots(2,1,figsize=(10,5.5),layout='constrained')
    for channel in range(n):
        axes[0].plot(time,values[:,channel],lw=.5,alpha=.8,label=f'Channel {channel+1}')
    axes[0].set(ylabel='Raw observation', xlabel='Retained index (zero based)')
    axes[0].legend(loc='upper right',ncol=n)
    if truth['kind'] == 'drifting':
        axes[1].plot(time,truth['coefficients'],lw=1)
        axes[1].axvline(truth['jump_index'],color='black',ls=':',lw=1,label='Jump to 0.8')
        axes[1].set(ylabel='Generating AR(1) coefficient')
        axes[1].legend()
    else:
        orders = np.full(N,4,dtype=int)
        if truth['kind'] == 'switching':
            for regime in truth['regimes']:
                orders[regime['start_index']:regime['stop_index']] = regime['order']
            for boundary in truth['switch_indices']:
                axes[0].axvline(boundary,color='black',lw=.5,alpha=.25)
        axes[1].step(time,orders,where='post',lw=1)
        axes[1].set(ylabel='Generating AR order',yticks=[1,2,3,4,6,8],ylim=(.5,8.5))
    axes[1].set(xlabel='Retained index (zero based)')
    title = f'{truth["kind"].capitalize()} synthetic data | n={n} | seed={seed}'
    if truth['kind']=='switching':
        title += f' | T={truth["regime_length"]}'
    fig.suptitle(title)
    fig.savefig(directory/'data_overview.pdf')
    plt.close(fig)


def cases():
    """Six stationary/switching settings, plus the scalar drifting example."""
    for n in (1,3):
        yield 'stationary',n,None
        for T in (500,2000):
            yield 'switching',n,T
    yield 'drifting',1,None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds',nargs='+',type=int,default=[685,686,687])
    parser.add_argument('--data-output',type=Path,default=Path('data/generated'))
    parser.add_argument('--output',type=Path,default=Path('outputs/synthetic_data'))
    parser.add_argument('--no-plots',action='store_true',help='Save data and checks without making figures')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds)<0:
        parser.error('Use distinct, nonnegative integer seeds')
    checks = []
    for kind,n,T in cases():
        case_path = Path(kind)/f'n{n}'
        if T is not None:
            case_path /= f'T{T}'
        for seed in args.seeds:
            cfg = DataConfig(seed=seed,channels=n)
            if kind=='stationary':
                values,truth = generate_stationary_ar4(cfg)
            elif kind=='switching':
                values,truth = generate_switching_ar(cfg,T)
            else:
                values,truth = generate_drifting_ar1(cfg)
            check = check_series(values,truth)
            destination = args.data_output/case_path/f'seed_{seed}'
            destination.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(destination/'observations.npz',observations_raw=values)
            write_json(destination/'ground_truth.json',truth)
            check['data_directory'] = destination.as_posix()
            checks.append(check)
            # Use the first listed seed for every figure, not a selected good-looking run.
            if seed==args.seeds[0] and not args.no_plots:
                plot_data(args.output/case_path/f'seed_{seed}',values,truth,seed)
        print(f'Generated and checked {case_path.as_posix()} for seeds {args.seeds}',flush=True)
    write_json(args.output/'generation_checks.json',{
        'seeds':args.seeds,'dataset_count':len(checks),'python':platform.python_version(),
        'numpy':np.__version__,'checks':checks,
        'plots_use_seed': None if args.no_plots else args.seeds[0],
        'scope':'Raw synthetic observations only; no expert, gate or baseline forecasts',
        'drifting_initialization':'200 discarded steps at coefficient 0.9; no reset at the halfway jump',
    })
    print(f'Saved {len(checks)} datasets under {args.data_output.resolve()}',flush=True)
    print(f'Checks and data figures: {args.output.resolve()}',flush=True)


if __name__=='__main__':
    main()
