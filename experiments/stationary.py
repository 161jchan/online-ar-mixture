"""Save results and make plots so we can inspect the generated data,
AR coefficient estimates, forecast errors, and failed fits.
"""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import platform
import numpy as np
from data.synthetic import DataConfig, generate_stationary_ar4, spectral_radius
from experts.online import ExpertConfig, OnlineARExperts
from experts.scalar import direct_yule_walker


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def run_stream(values, config):
    """Run AR(1)..AR(L) through the observations and save their forecasts.

    This function receives observations and settings, but not the real
    AR(4) coefficients. It predicts each target before passing that target
    to update(). The default maximum order is L=8.
    """
    bank = OnlineARExperts(config).initialize(values[:config.warmup_length])
    fields = {name: [] for name in (
        'forecasts_standardized', 'forecasts_raw', 'targets_standardized',
        'targets_raw', 'coefficients', 'error_variances_standardized',
        'covariances_standardized', 'fresh', 'stale_steps', 'reflections',
        'failed_order', 'expert_spectral_radii')}
    coefficient_error = variance_error = 0.0
    comparisons = 0
    for target_index in range(config.warmup_length, len(values)):
        forecasts = bank.predict()
        snapshot = bank.snapshot()
        if snapshot['next_target_index'] != target_index:
            raise AssertionError('Forecast/target alignment failed')
        # At every scored step, compare each successfully fitted AR order
        # with the direct NumPy solution using the same covariance estimates.
        for p in np.flatnonzero(snapshot['diagnostics']['fresh']) + 1:
            a, v = direct_yule_walker(snapshot['covariances_standardized'], p,
                                     epsilon=config.epsilon)
            error = np.max(np.abs(a - snapshot['coefficients'][p-1, :p]))
            coefficient_error = max(coefficient_error, float(error))
            variance_error = max(variance_error,
                                 abs(v-snapshot['error_variances_standardized'][p-1]))
            comparisons += 1
        # Save each expert's forecast and coefficients alongside the actual target
        # before updating AR(1)..AR(L) for the following observation.
        # After the loop, calculate squared errors from the saved forecasts and targets.
        target = values[target_index]
        fields['forecasts_standardized'].append(forecasts)
        fields['forecasts_raw'].append(bank.inverse_transform(forecasts))
        fields['targets_standardized'].append(bank.standardize(target))
        fields['targets_raw'].append(target.copy())
        for key in ('coefficients', 'error_variances_standardized', 'covariances_standardized'):
            fields[key].append(snapshot[key])
        for key in ('fresh', 'stale_steps'):
            fields[key].append(snapshot['diagnostics'][key])
        fields['reflections'].append(snapshot['diagnostics']['reflections_current_attempt'])
        fields['failed_order'].append(snapshot['diagnostics']['failed_order'] or 0)
        fields['expert_spectral_radii'].append([
            spectral_radius(snapshot['coefficients'][p-1, :p])
            for p in range(1, config.max_order + 1)])
        bank.update(target)
    arrays = {key: np.asarray(value) for key, value in fields.items()}
    arrays['target_indices'] = np.arange(config.warmup_length, len(values))
    arrays['squared_errors_standardized'] = (
        arrays['targets_standardized'][:, None, :] - arrays['forecasts_standardized'])**2
    arrays['squared_errors_raw'] = (
        arrays['targets_raw'][:, None, :] - arrays['forecasts_raw'])**2
    if not np.isfinite(arrays['forecasts_standardized']).all():
        raise AssertionError('Nonfinite forecasts')
    if coefficient_error > 1e-9 or variance_error > 1e-9:
        raise AssertionError('Levinson/direct solve mismatch')
    return arrays, bank.snapshot(), {
        'valid_order_comparisons': comparisons,
        'max_absolute_coefficient_difference': coefficient_error,
        'max_absolute_variance_difference': variance_error,
        'comparison_tolerance': 1e-9,
    }


def empirical_acf(values, max_lag=32):
    centered = values - values.mean()
    denominator = centered @ centered
    if denominator == 0:
        return np.r_[1., np.zeros(max_lag)]
    return np.array([centered[k:] @ centered[:len(centered)-k] / denominator
                     for k in range(max_lag + 1)])


def plot_diagnostics(directory, values, truth, arrays, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 130})

    def save(fig, name):
        fig.savefig(directory / f'{name}.png', dpi=150)
        fig.savefig(directory / f'{name}.pdf')
        plt.close(fig)

    t = arrays['target_indices']
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.3), layout='constrained')
    axes[0].plot(np.arange(len(values)), values[:, 0], lw=.5)
    axes[0].axvspan(0, 39, color='orange', alpha=.4, label='warm-up only')
    axes[0].set(xlabel='Retained observation index (zero based)', ylabel='Raw x',
                title=f'Stationary AR(4), seed {summary["seed"]}')
    axes[0].legend(loc='upper right')
    acf = empirical_acf(values[:, 0])
    axes[1].stem(np.arange(len(acf)), acf, basefmt=' ')
    axes[1].set(xlabel='Lag', ylabel='Descriptive full-series ACF',
                title='Lag dependence (not used for preprocessing)')
    save(fig, 'data_diagnostics')

    # Compare the estimated lag-1..4 coefficients from AR(4)..AR(8)
    # with the real generating coefficients. This is done after forecasting.
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout='constrained')
    colors = plt.cm.viridis(np.linspace(.1, .9, 5))
    for lag, ax in enumerate(axes.flat):
        for color, p in zip(colors, range(4, min(8, arrays['coefficients'].shape[1])+1)):
            ax.plot(t[::10], arrays['coefficients'][::10, p-1, lag],
                    color=color, lw=.65, alpha=.85, label=f'AR({p})')
        ax.axhline(truth['coefficients'][lag], color='black', ls='--', lw=1.2, label='Truth')
        ax.set(xlabel='Next target index', ylabel=f'Lag {lag+1} coefficient')
    axes[0, 0].legend(ncol=3, fontsize=8)
    fig.suptitle('Coefficient paths for orders 4 and above (retained estimates included)')
    save(fig, 'coefficient_paths')

    L = arrays['coefficients'].shape[1]
    orders = np.arange(1, L+1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout='constrained')
    means, spreads = [], []
    for p in range(L):
        x = arrays['error_variances_standardized'][-2000:, p]
        fresh = arrays['fresh'][-2000:, p]
        means.append(float(x[fresh].mean()) if fresh.any() else np.nan)
        spreads.append(float(x[fresh].std()) if fresh.any() else np.nan)
    axes[0].errorbar(orders, means, yerr=spreads, marker='o', capsize=3,
                     label='Last 2,000: mean +/- temporal SD (fresh only)')
    axes[0].plot(orders, arrays['error_variances_standardized'][-1], 'x--', label='Last pre-target fit')
    axes[0].axhline(summary['noise_variance_standardized'], color='black', ls=':', label='Innovation variance / warm-up scale squared')
    axes[0].set(xlabel='Expert order', ylabel='Standardized prediction-error variance', xticks=orders)
    axes[0].legend(fontsize=7)
    axes[1].plot(orders, summary['expert_mse_raw'], 'o-')
    axes[1].axhline(1, color='black', ls=':', label='Raw innovation variance = 1')
    axes[1].set(xlabel='Expert order', ylabel='Raw one-step MSE (warm-up excluded)', xticks=orders)
    axes[1].legend(fontsize=8)
    fig.suptitle('Model variance estimates and realized forecast loss are different quantities')
    save(fig, 'variance_by_order')

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), layout='constrained')
    axes[0].plot(t[:160], arrays['targets_raw'][:160, 0], label='Target', lw=1)
    axes[0].plot(t[:160], arrays['forecasts_raw'][:160, min(3,L-1), 0], label=f'AR({min(4,L)}) forecast', lw=1)
    axes[0].set(xlabel='Target index (same index for forecast and target)', ylabel='Raw x')
    axes[0].legend(ncol=2)
    axes[1].step(t, arrays['failed_order'], lw=.6)
    axes[1].set(xlabel='Next target index', ylabel='First failed order (0 = none)')
    axes[2].imshow(arrays['stale_steps'].T > 0, origin='lower', aspect='auto',
                   interpolation='nearest', extent=[t[0], t[-1], .5, L+.5],
                   cmap='Greys', vmin=0, vmax=1)
    axes[2].set(xlabel='Next target index', ylabel='Expert order', yticks=orders,
                title='Black marks retained/stale estimates')
    save(fig, 'alignment_and_guards')


def run_one(root, data_root, seed, config):
    directory = root / f'seed_{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    data_directory = data_root / f'seed_{seed}'
    data_directory.mkdir(parents=True, exist_ok=True)
    values, truth = generate_stationary_ar4(DataConfig(seed=seed))
    arrays, final, direct = run_stream(values, config)
    normalization = final['normalization']
    scale = normalization['scale']
    finite = np.isfinite(values)
    summary = {
        'seed': seed, 'n_observations': len(values), 'warmup_length': config.warmup_length,
        'scored_observations': len(arrays['target_indices']),
        'nonfinite_observations': int((~finite).sum()),
        'distribution_raw': {'mean': float(values.mean()), 'std_ddof0': float(values.std()),
                             'min': float(values.min()), 'max': float(values.max()),
                             'quantiles_01_25_50_75_99': np.quantile(values, [.01,.25,.5,.75,.99]).tolist()},
        'spectral_radius_truth': truth['spectral_radius'],
        'normalization': normalization, 'noise_variance_raw': 1.0,
        'noise_variance_standardized': 1 / scale**2,
        'expert_mse_raw': arrays['squared_errors_raw'].mean(axis=(0, 2)).tolist(),
        'expert_mse_standardized': arrays['squared_errors_standardized'].mean(axis=(0, 2)).tolist(),
        'guard_events_including_final_fit': final['diagnostics']['guard_events'],
        'scored_states_with_guard': int(np.count_nonzero(arrays['failed_order'])),
        'stale_fraction_by_order': (~arrays['fresh']).mean(axis=0).tolist(),
        'max_stale_steps_by_order': arrays['stale_steps'].max(axis=0).tolist(),
        'all_forecasts_finite': bool(np.isfinite(arrays['forecasts_raw']).all()),
        'max_expert_spectral_radius': float(arrays['expert_spectral_radii'].max()),
        'forecast_alignment': 'predict at index t from observations 0..t-1; t=5L..9999',
        'direct_yule_walker': direct,
        'last_2000_mean_error_variance_standardized_fresh_only': [
            float(arrays['error_variances_standardized'][-2000:, p][arrays['fresh'][-2000:, p]].mean())
            if arrays['fresh'][-2000:, p].any() else None for p in range(config.max_order)],
        'interpretation': [
            'Full-series summaries and ACF are descriptive only; normalization is warm-up-only.',
            'With fixed forgetting, coefficients fluctuate; exact convergence is not expected.',
            'Fresh variance estimates decrease within a valid recursion; stale mixed-time orders need not.',
            'Warm-up mean is an estimate: frozen centering can leave residual mean bias.',
            'Temporal SD bars describe fluctuations and are not confidence intervals.',
        ],
    }
    np.savez_compressed(data_directory / 'observations.npz', observations_raw=values)
    np.savez_compressed(directory / 'online_diagnostics.npz', **arrays)
    write_json(data_directory / 'ground_truth.json', truth)
    write_json(directory / 'run_configuration.json', {'data': truth['configuration'], 'experts': asdict(config),
                                                     'python': platform.python_version(), 'numpy': np.__version__})
    write_json(directory / 'summary.json', summary)
    with (directory / 'expert_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['order', 'mse_raw', 'mse_standardized', 'stale_fraction'])
        writer.writerows(zip(range(1, config.max_order+1), summary['expert_mse_raw'],
                            summary['expert_mse_standardized'], summary['stale_fraction_by_order']))
    plot_diagnostics(directory, values, truth, arrays, summary)
    return summary, truth


def write_report_results(path, summaries, first_truth, figure_root):
    """If requested, export the seeds, real generating coefficients, stability
    result, normalization settings, scored observation count, solver-check
    difference, failed-fit count, and per-expert MSE table for LaTeX.

    The complete estimated coefficient paths remain in online_diagnostics.npz.
    """
    first = summaries[0]
    mse = np.array([s['expert_mse_raw'] for s in summaries])
    a = first_truth['coefficients']
    content = [
        '% Generated by python -m experiments.stationary; do not edit numbers by hand.',
        '\\newcommand{\\ARSeeds}{' + ', '.join(str(s['seed']) for s in summaries) + '}',
        '\\newcommand{\\ARFigureDirectory}{' + (figure_root / f'seed_{first["seed"]}').as_posix() + '}',
        '\\newcommand{\\ARTrueCoefficients}{' + ', '.join(f'{v:.6f}' for v in a) + '}',
        f'\\newcommand{{\\ARRadius}}{{{first["spectral_radius_truth"]:.6f}}}',
        f'\\newcommand{{\\ARMean}}{{{first["normalization"]["mean"]:.6f}}}',
        f'\\newcommand{{\\ARScale}}{{{first["normalization"]["scale"]:.6f}}}',
        f'\\newcommand{{\\ARNoiseZ}}{{{first["noise_variance_standardized"]:.6f}}}',
        f'\\newcommand{{\\ARScored}}{{{first["scored_observations"]}}}',
        '\\newcommand{\\ARMaxDirectDifference}{' + f'{max(s["direct_yule_walker"]["max_absolute_coefficient_difference"] for s in summaries):.3g}' + '}',
        '\\newcommand{\\ARTotalGuards}{' + str(sum(s['guard_events_including_final_fit'] for s in summaries)) + '}',
        '\\newcommand{\\ARResultsTable}{%',
        '\\begin{tabular}{rcc}\\hline',
        'Order & Raw MSE mean & Between-seed SD \\\\ \\hline',
    ]
    for p in range(mse.shape[1]):
        sd = f'{mse[:, p].std(ddof=1):.5f}' if len(summaries) > 1 else '---'
        content.append(f'{p+1} & {mse[:,p].mean():.5f} & {sd} \\\\')
    content += ['\\hline\\end{tabular}}', '']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(content), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=[685, 686, 687])
    parser.add_argument('--max-order', type=int, default=8)
    parser.add_argument('--forgetting-factor', type=float, default=.99)
    parser.add_argument('--epsilon', type=float, default=1e-6)
    parser.add_argument('--output', type=Path, default=Path('outputs/stationary/n1'))
    parser.add_argument('--data-output', type=Path, default=Path('data/generated/stationary/n1'),
                        help='Save generated observations and real coefficients here')
    parser.add_argument('--report-results', type=Path,
                        help='Optional LaTeX numeric export; no report folder is created by default')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Use distinct seeds')
    config = ExpertConfig(max_order=args.max_order, forgetting_factor=args.forgetting_factor,
                          epsilon=args.epsilon)
    summaries = []
    first_truth = None
    for seed in args.seeds:
        summary, truth = run_one(args.output, args.data_output, seed, config)
        summaries.append(summary)
        first_truth = truth if first_truth is None else first_truth
        print(f'Seed {seed}: {summary["scored_observations"]} scored; '
              f'AR(4) raw MSE={summary["expert_mse_raw"][min(3,args.max_order-1)]:.5f}; '
              f'guards={summary["guard_events_including_final_fit"]}', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'aggregate_summary.json', {
        'seeds': args.seeds,
        'expert_mse_raw_mean': np.mean([s['expert_mse_raw'] for s in summaries], axis=0).tolist(),
        'expert_mse_raw_sample_sd': np.std([s['expert_mse_raw'] for s in summaries], axis=0,
                                          ddof=1 if len(summaries)>1 else 0).tolist(),
        'single_seed_sd_defined': len(summaries)>1,
        'runs': summaries,
    })
    if args.report_results is not None:
        write_report_results(args.report_results, summaries, first_truth, args.output)
    print(f'Generated data: {args.data_output.resolve()}')
    print(f'Diagnostics: {args.output.resolve()}')


if __name__ == '__main__':
    main()
