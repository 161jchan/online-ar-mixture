"""Run Jenny's experts/FPE on cached observations; redraw PDF figures offline."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time
import numpy as np
from data.real import load_real, sha256
from experts.online import ExpertConfig, OnlineARExperts
from experts.fpe import select_fpe
from experts.multichannel import direct_yule_walker as direct_vector
from experts.scalar import direct_yule_walker as direct_scalar

ROOT = Path("outputs/experts_fpe")
LAMBDAS = (.98, .99, .995)


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, default=serial, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def verify_fit(snapshot, epsilon):
    """Check every currently fresh order independently against the same block system."""
    r = snapshot["covariances_standardized"]
    a = snapshot["coefficients_standardized"]
    sigma = snapshot["error_covariances_standardized"]
    maximum = 0.
    count = 0
    for i in np.flatnonzero(snapshot["diagnostics"]["fresh"]):
        if snapshot["channels"] == 1:
            direct_a, direct_s = direct_scalar(r, int(i+1), epsilon=epsilon)
            direct_a, direct_s = direct_a[:, None, None], np.array([[direct_s]])
        else:
            direct_a, direct_s = direct_vector(r, int(i+1), epsilon=epsilon)
        np.testing.assert_allclose(a[i, :i+1], direct_a, atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(sigma[i], direct_s, atol=1e-8, rtol=1e-8)
        maximum = max(maximum, float(np.max(abs(a[i, :i+1]-direct_a))),
                      float(np.max(abs(sigma[i]-direct_s))))
        count += 1
    return maximum, count


def run_stream(values, config, *, observed=None, save_arrays=True, verify=True):
    """Receive observations only. Generating coefficients never enter this loop."""
    x = np.asarray(values, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n, L, start = x.shape[1], config.max_order, config.warmup_length
    if len(x) <= start:
        raise ValueError("Stream must extend beyond 5L warm-up observations")
    observed = np.ones_like(x, bool) if observed is None else np.asarray(observed, bool)
    if observed.shape != x.shape:
        raise ValueError("Observed mask must match observations")
    bank = OnlineARExperts(config).initialize(x[:start])
    length = len(x)-start
    arrays = {}
    shapes = {
        "forecasts_standardized": (length, L, n), "forecasts_raw": (length, L, n),
        "error_covariances_standardized": (length, L, n, n),
        "error_covariances_raw": (length, L, n, n),
        "fpe_forecasts_standardized": (length, n), "fpe_forecasts_raw": (length, n),
        "fresh": (length, L), "stale_steps": (length, L),
        "fpe_log_scores": (length, L), "fpe_order": (length,),
        "fpe_fallback": (length,), "normalized_reflection_norms": (length, L),
        "failed_order": (length,),
    }
    if save_arrays:
        for key, shape in shapes.items():
            dtype = bool if key in ("fresh", "fpe_fallback") else (
                np.int32 if key in ("stale_steps", "fpe_order", "failed_order") else float)
            arrays[key] = np.empty(shape, dtype=dtype)
    sum_z, sum_raw, fpe_z, fpe_raw = np.zeros(L), np.zeros(L), 0., 0.
    counts = np.zeros(L, int)
    fallbacks = scored = guard_states = 0
    stale_counts, max_stale = np.zeros(L), np.zeros(L, int)
    hist = np.zeros(L, int)
    max_difference = 0.
    comparisons = 0
    previous_order = 1
    path_indices, paths_z, paths_raw = [], [], []
    coefficient_stride = 10 if n <= 3 else 25
    # Snapshots are always taken before the corresponding target is observed.
    for j, t in enumerate(range(start, len(x))):
        forecast_z = bank.predict()
        s = bank.snapshot()
        if s["next_target_index"] != t:
            raise AssertionError("Forecast/target index mismatch")
        diag = s["diagnostics"]
        if verify:
            difference, checked = verify_fit(s, config.epsilon)
            max_difference = max(max_difference, difference)
            comparisons += checked
        choice = select_fpe(s["error_covariances_standardized"], config.forgetting_factor,
                            fresh=diag["fresh"], previous_order=previous_order)
        previous_order = choice.order
        forecast_raw = bank.inverse_transform(forecast_z)
        target_z = bank.standardize(x[t])
        if observed[t].all():
            sum_z += ((forecast_z-target_z)**2).sum(axis=1)
            sum_raw += ((forecast_raw-x[t])**2).sum(axis=1)
            fpe_z += float(((forecast_z[choice.order-1]-target_z)**2).sum())
            fpe_raw += float(((forecast_raw[choice.order-1]-x[t])**2).sum())
            scored += 1
            hist[choice.order-1] += 1
            fallbacks += int(choice.used_fallback)
        stale_counts += ~diag["fresh"]
        max_stale = np.maximum(max_stale, diag["stale_steps"])
        counts += diag["fresh"]
        guard_states += int(diag["failed_order"] is not None)
        if save_arrays:
            for key, value in (
                ("forecasts_standardized", forecast_z), ("forecasts_raw", forecast_raw),
                ("fpe_forecasts_standardized", forecast_z[choice.order-1]),
                ("fpe_forecasts_raw", forecast_raw[choice.order-1]),
                ("fpe_log_scores", choice.log_scores), ("fpe_order", choice.order),
                ("fpe_fallback", choice.used_fallback),
                ("failed_order", diag["failed_order"] or 0)):
                arrays[key][j] = value
            for key in ("error_covariances_standardized", "error_covariances_raw"):
                arrays[key][j] = s[key]
            for key in ("fresh", "stale_steps", "normalized_reflection_norms"):
                arrays[key][j] = diag[key]
            if j % coefficient_stride == 0:
                path_indices.append(t)
                paths_z.append(s["coefficients_standardized"])
                paths_raw.append(s["coefficients_raw"])
        bank.update(x[t])
    if not scored:
        raise ValueError("No fully observed targets after warm-up")
    summary = {
        "configuration": asdict(config), "channels": n, "rows": len(x),
        "warmup_length": start, "scored_observations": scored,
        "excluded_incomplete_targets": length-scored,
        "expert_mse_standardized": sum_z/scored, "expert_mse_raw": sum_raw/scored,
        "fpe_mse_standardized": fpe_z/scored, "fpe_mse_raw": fpe_raw/scored,
        "fpe_order_counts": hist, "fpe_fallback_scored_steps": fallbacks,
        "stale_fraction_by_order": stale_counts/length, "max_stale_steps_by_order": max_stale,
        "scored_states_with_guard": guard_states, "fresh_counts": counts,
        "direct_solve": {"valid_order_comparisons": comparisons,
                         "maximum_absolute_difference": max_difference,
                         "atol": 1e-8, "rtol": 1e-8},
        "normalization": bank.snapshot()["normalization"],
        "loss_units": "Mean squared Euclidean norm, summed across channels; not divided by n",
        "timing": "Row t is forecast with observations [0,t); covariance/fresh/FPE also pre-target",
    }
    if save_arrays:
        arrays.update(
            target_indices=np.arange(start, len(x)), targets_raw=x[start:],
            targets_standardized=bank.standardize(x[start:]), observed_mask=observed[start:],
            scoring_mask=observed[start:].all(axis=1), coefficient_target_indices=np.array(path_indices),
            coefficients_standardized=np.array(paths_z), coefficients_raw=np.array(paths_raw),
            normalization_mean=np.asarray(summary["normalization"]["mean"]),
            normalization_scale=np.asarray(summary["normalization"]["scale"]),
        )
    return arrays, summary


def synthetic_paths():
    return sorted(Path("data/generated").rglob("observations.npz"))


def run_synthetic(root, *, variants=False):
    for path in synthetic_paths():
        relative = path.parent.relative_to(Path("data/generated"))
        destination = root/"synthetic"/relative
        configs = [(8, .99)]
        # Small factorial slices; keep T=2000 and drift main runs only.
        if variants and ("stationary" in relative.parts or "T500" in relative.parts):
            configs += [(4, .99), (12, .99), (8, .98), (8, .995)]
        x = np.load(path)["observations_raw"]
        for L, lam in configs:
            main = (L, lam) == (8, .99)
            directory = destination if main else destination/f"L{L}_lambda{lam:g}"
            if (directory/"summary.json").exists():
                cached = json.loads((directory/"summary.json").read_text())
                if cached["source_sha256"] != sha256(path):
                    raise ValueError(f"Saved observations changed after run: {path}")
                print("Cached", directory, flush=True)
                continue
            arrays, summary = run_stream(x, ExpertConfig(max_order=L, forgetting_factor=lam),
                                          save_arrays=main, verify=main)
            summary.update(dataset=relative.as_posix(), source_sha256=sha256(path),
                           source_file=path.as_posix(), synthetic_seed=int(relative.parts[-1].split("_")[-1]))
            directory.mkdir(parents=True, exist_ok=True)
            if main:
                np.savez_compressed(directory/"forecasts.npz", **arrays)
            write_json(directory/"summary.json", summary)
            print(relative, "L", L, "lambda", lam, "FPE", round(summary["fpe_mse_raw"], 5), flush=True)


def run_real(root):
    for name in ("etth1", "sunspots"):
        x, timestamps, observed, metadata = load_real(name)
        directory = root/"real"/name
        if (directory/"summary.json").exists():
            print("Cached", directory, flush=True)
            continue
        split = int(.2*len(x))
        protocol_path = directory/"frozen_protocol.json"
        if protocol_path.exists():
            protocol = json.loads(protocol_path.read_text())
            if protocol["data"]["sha256"] != metadata["sha256"]:
                raise ValueError("Data changed after protocol was frozen")
        else:
            trials = []
            for lam in LAMBDAS:
                _, summary = run_stream(x[:split], ExpertConfig(forgetting_factor=lam),
                                        observed=observed[:split], save_arrays=False, verify=True)
                trials.append({"forgetting_factor": lam,
                               "prefix_fpe_mse_standardized": summary["fpe_mse_standardized"]})
            selected = min(trials, key=lambda r: r["prefix_fpe_mse_standardized"])["forgetting_factor"]
            protocol = {
                "data": metadata, "split_index": split, "first20_percent_trials": trials,
                "selected_forgetting_factor": selected, "max_order": 8,
                "selection_rule": "Lowest prefix prequential FPE standardized loss; ties use grid order",
                "heldout_policy": "Freeze this file before the first full pass. No retuning after index split.",
                "deterministic_experts": "No gate initialization; repeating identical real experts for three seeds adds no variation",
            }
            write_json(protocol_path, protocol)
        arrays, summary = run_stream(x, ExpertConfig(forgetting_factor=protocol["selected_forgetting_factor"]),
                                     observed=observed, save_arrays=True, verify=True)
        indices = arrays["target_indices"]
        arrays["timestamps"] = timestamps[indices]
        heldout = (indices >= split) & arrays["scoring_mask"]
        for suffix in ("raw", "standardized"):
            target = arrays[f"targets_{suffix}"]
            error = ((arrays[f"forecasts_{suffix}"]-target[:, None])**2).sum(axis=2)
            fpe_error = ((arrays[f"fpe_forecasts_{suffix}"]-target)**2).sum(axis=1)
            summary[f"heldout_expert_mse_{suffix}"] = error[heldout].mean(axis=0)
            summary[f"heldout_fpe_mse_{suffix}"] = float(fpe_error[heldout].mean())
        summary.update(dataset=name, data=metadata, split_index=split,
                       heldout_scored_observations=int(heldout.sum()), full_stream_descriptive_only=True)
        np.savez_compressed(directory/"forecasts.npz", **arrays)
        write_json(directory/"summary.json", summary)
        print(name, "held-out FPE", summary["heldout_fpe_mse_standardized"], flush=True)


def plot_one(directory):
    """Figures consume saved forecasts; no predictor or real-data tuning is run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summary = json.loads((directory/"summary.json").read_text())
    a = np.load(directory/"forecasts.npz")
    t, n = a["target_indices"], summary["channels"]
    L = a["forecasts_raw"].shape[1]
    loss = ((a["forecasts_standardized"]-a["targets_standardized"][:, None])**2).sum(axis=2)
    fpe_loss = ((a["fpe_forecasts_standardized"]-a["targets_standardized"])**2).sum(axis=1)
    mask = a["scoring_mask"]
    if "split_index" in summary:
        mask &= t >= summary["split_index"]
    valid_indices = np.flatnonzero(mask)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), layout="constrained")
    for order in (1, min(4, L), L):
        axes[0].plot(t[valid_indices], np.cumsum(loss[mask, order-1])/np.arange(1, mask.sum()+1),
                     lw=.8, label=f"AR({order})")
    axes[0].plot(t[valid_indices], np.cumsum(fpe_loss[mask])/np.arange(1, mask.sum()+1),
                 color="black", lw=1, label="FPE")
    axes[0].set(ylabel=("Held-out cumulative MSE\n(standardized)" if "split_index" in summary
                        else "Cumulative standardized loss"), title=summary["dataset"])
    axes[0].legend(ncol=4)
    axes[1].step(t, a["fpe_order"], lw=.6)
    axes[1].set(ylabel="FPE selected AR order", yticks=np.arange(1, L+1))
    axes[2].imshow(~a["fresh"].T, aspect="auto", origin="lower", cmap="Greys",
                   extent=(t[0], t[-1], .5, L+.5), interpolation="nearest", vmin=0, vmax=1)
    axes[2].set(xlabel="Next target index (zero based)", ylabel="AR order",
                title="Black = retained/stale fit; all forecasts remain available")
    fig.savefig(directory/"expert_fpe.pdf")
    plt.close(fig)
    if "/stationary/" not in directory.as_posix():
        return
    truth_path = Path(summary["source_file"]).parent/"ground_truth.json"
    truth = np.asarray(json.loads(truth_path.read_text())["coefficients"]).reshape(4, n, n)
    estimated = a["coefficients_raw"]
    # Every entry of all four AR(4) lag matrices is shown in raw coefficient units.
    fig, axes = plt.subplots(n, n, figsize=(10, 2.4*n+1), squeeze=False, layout="constrained")
    colors = plt.cm.tab10(np.arange(4))
    for out in range(n):
        for inp in range(n):
            ax = axes[out, inp]
            for k, color in enumerate(colors):
                ax.plot(a["coefficient_target_indices"], estimated[:, 3, k, out, inp],
                        color=color, lw=.65, label=f"Lag {k+1}")
                ax.axhline(truth[k, out, inp], color=color, ls="--", lw=1)
            ax.set(xlabel="Next target index", ylabel=f"A[k][{out},{inp}] raw")
    axes[0, 0].legend(ncol=2, fontsize=7)
    fig.suptitle("AR(4) coefficient entries; dashed = generating coefficient; retained fits included")
    fig.savefig(directory/"coefficient_paths.pdf")
    plt.close(fig)

    padded_truth = np.zeros((L, n, n))
    padded_truth[:4] = truth
    coefficient_errors = np.sqrt(np.sum((estimated-padded_truth[None, None])**2, axis=(2, 3, 4)))
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    tail = coefficient_errors[-200:]
    ax.errorbar(np.arange(1, L+1), tail.mean(axis=0), yerr=tail.std(axis=0),
                marker="o", capsize=3)
    ax.set(xlabel="AR order", ylabel="Frobenius norm of all raw lag-coefficient errors",
           title="Last 200 saved coefficient fits: mean +/- temporal SD")
    fig.savefig(directory/"coefficients_by_order.pdf")
    plt.close(fig)


    sigma = a["error_covariances_raw"]
    # Symmetric covariance entries, not an unlabeled matrix norm.
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    fig, axes = plt.subplots(len(pairs), 1, figsize=(10, 1.6*len(pairs)+1),
                             squeeze=False, layout="constrained")
    for ax, (i, j) in zip(axes[:, 0], pairs):
        for p in (1, 4, L):
            ax.plot(t[::10], sigma[::10, p-1, i, j], lw=.65, label=f"AR({p})")
        ax.axhline(float(i == j), color="black", ls="--", label="Innovation covariance")
        ax.set(ylabel=f"Sigma[{i},{j}] raw", xlabel="Next target index")
    axes[0, 0].legend(ncol=4, fontsize=7)
    fig.suptitle("Prediction-error covariance entries over time (retained estimates included)")
    fig.savefig(directory/"covariance_paths.pdf")
    plt.close(fig)
    trace = np.trace(sigma, axis1=2, axis2=3)/n
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    tail = np.arange(len(t)) >= max(0, len(t)-2000)
    means, spreads = [], []
    for p in range(L):
        eligible = tail & a["fresh"][:, p]
        means.append(trace[eligible, p].mean() if eligible.any() else np.nan)
        spreads.append(trace[eligible, p].std() if eligible.any() else np.nan)
    axes[0].errorbar(np.arange(1, L+1), means, yerr=spreads, marker="o", capsize=3)
    axes[0].axhline(1, color="black", ls="--")
    axes[0].set(xlabel="AR order", ylabel="Raw trace(Sigma_p)/n",
                title="Last 2,000 fresh fits: mean +/- temporal SD")
    axes[1].plot(np.arange(1, L+1), np.mean(
        (a["forecasts_raw"]-a["targets_raw"][:, None])**2, axis=(0, 2)), "o-")
    axes[1].axhline(1, color="black", ls="--")
    axes[1].set(xlabel="AR order", ylabel="Raw one-step loss / channel",
                title="All scored targets; warm-up excluded")
    fig.savefig(directory/"covariance_by_order.pdf")
    plt.close(fig)



def common_interval_loss(summary):
    """Remove the short pre-60 prefix for fair L=4,8,12 comparisons.

    Only at most 40 already-used synthetic targets are replayed; the complete
    stream and its normalization are unchanged. No real held-out data enter.
    """
    config = ExpertConfig(**summary["configuration"])
    x = np.load(summary["source_file"])["observations_raw"]
    start, stop = config.warmup_length, 60
    prefix_expert = np.zeros(config.max_order)
    prefix_fpe = 0.
    if start < stop:
        bank = OnlineARExperts(config).initialize(x[:start])
        previous = 1
        for target in x[start:stop]:
            forecasts = bank.inverse_transform(bank.predict())
            s = bank.snapshot()
            choice = select_fpe(s["error_covariances_standardized"], config.forgetting_factor,
                                fresh=s["diagnostics"]["fresh"], previous_order=previous)
            previous = choice.order
            losses = ((forecasts-target)**2).sum(axis=1)
            prefix_expert += losses
            prefix_fpe += float(losses[choice.order-1])
            bank.update(target)
    total = summary["scored_observations"]
    return ((np.asarray(summary["expert_mse_raw"])*total-prefix_expert)/(len(x)-stop),
            (summary["fpe_mse_raw"]*total-prefix_fpe)/(len(x)-stop))


def plot_comparisons(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = json.loads((root/"aggregate.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    prefixes = ("stationary/n1", "stationary/n3", "switching/n1/T500", "switching/n3/T500")
    configs = ((4, .99), (8, .99), (12, .99), (8, .98), (8, .995))
    labels = [f"L={L}\n"+rf"$\lambda={lam:g}$" for L, lam in configs]
    for ax, prefix in zip(axes.flat, prefixes):
        rows = [data.get(f"{prefix}/L{L}_lambda{lam:g}") for L, lam in configs]
        if not all(rows):
            ax.set_visible(False)
            continue
        n = 3 if "/n3" in prefix else 1
        ax.errorbar(np.arange(5), [r["common_interval_fpe_mse_raw_mean"]/n for r in rows],
                    yerr=[r["common_interval_fpe_mse_raw_sample_sd"]/n for r in rows],
                    marker="o", capsize=3)
        ax.set(xticks=np.arange(5), xticklabels=labels, ylabel="FPE raw loss / channel",
               title=prefix+"; mean +/- SD across seeds")
    fig.suptitle("L and forgetting comparisons: identical targets 60..9999; seeds 685, 686, 687")
    fig.savefig(root/"comparisons.pdf")
    plt.close(fig)


def aggregate(root):
    summaries = []
    for path in sorted(root.rglob("summary.json")):
        s = json.loads(path.read_text())
        s["summary_file"] = path.as_posix()
        summaries.append(s)
    grouped = {}
    for s in summaries:
        if "synthetic_seed" not in s:
            continue
        key = s["dataset"].rsplit("/seed_", 1)[0]
        c = s["configuration"]
        key += f'/L{c["max_order"]}_lambda{c["forgetting_factor"]:g}'
        grouped.setdefault(key, []).append(s)
    result = {}
    for key, runs in grouped.items():
        fpe = np.array([s["fpe_mse_raw"] for s in runs])
        common = np.array([common_interval_loss(s)[1] for s in runs])
        mse = np.array([s["expert_mse_raw"] for s in runs])
        result[key] = {
            "seeds": [s["synthetic_seed"] for s in runs], "fpe_mse_raw_mean": fpe.mean(),
            "comparison_target_start_index": 60,
            "common_interval_fpe_mse_raw_mean": common.mean(),
            "common_interval_fpe_mse_raw_sample_sd": common.std(ddof=1) if len(runs)>1 else None,
            "fpe_mse_raw_sample_sd": fpe.std(ddof=1) if len(runs)>1 else None,
            "expert_mse_raw_mean": mse.mean(axis=0),
            "expert_mse_raw_sample_sd": mse.std(axis=0, ddof=1) if len(runs)>1 else None,
            "mean_stale_fraction_by_order": np.mean([s["stale_fraction_by_order"] for s in runs], axis=0),
        }
    write_json(root/"aggregate.json", result)
    write_json(root/"verification.json", {
        "runs": len(summaries),
        "direct_valid_order_comparisons": sum(s["direct_solve"]["valid_order_comparisons"] for s in summaries),
        "maximum_direct_absolute_difference": max(s["direct_solve"]["maximum_absolute_difference"] for s in summaries),
        "human_multichannel_review": "PENDING: no teammate review has occurred",
        "software": {"python": platform.python_version(), "numpy": np.__version__},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("synthetic", "real", "plots", "aggregate"))
    parser.add_argument("--variants", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    started = time.perf_counter()
    if args.action == "synthetic":
        run_synthetic(args.output, variants=args.variants)
    elif args.action == "real":
        run_real(args.output)
    elif args.action == "plots":
        for path in sorted(args.output.rglob("forecasts.npz")):
            plot_one(path.parent)
        plot_comparisons(args.output)
    if args.action != "plots":
        aggregate(args.output)
    print("Seconds:", round(time.perf_counter()-started, 1), flush=True)


if __name__ == "__main__":
    main()
