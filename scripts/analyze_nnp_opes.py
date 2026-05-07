from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eatr_rates.dataset_config import OpesAnalysisConfig, load_opes_config
from eatr_rates.analysis_io import collect_completed_run_files, sorted_prefixed_dirs
from eatr_rates.plot_style import (
    BLACK,
    BLUE,
    GRAY,
    LIGHT_BLUE,
    SET_COLORS,
    add_panel_labels,
    apply_publication_style,
    style_axis,
    style_axes,
)
import ks_censored as ksc
from scripts.run_example_analyses import (
    bootstrap_flooding,
    flooding_diagnostics,
    flooding_log_average,
    thread_map,
)
import rate_methods_library as RM

DEFAULT_CONFIG_PATH = ROOT / "analysis-configs" / "nnp_opes.toml"
MPL_CONFIG_DIR = ROOT / ".matplotlib-cache"
XDG_CACHE_DIR = ROOT / ".cache"


def ensure_plot_env() -> None:
    MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
    os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))
    os.environ.setdefault("MPLBACKEND", "Agg")


def pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    apply_publication_style(plt)

    return plt


def beta_value(config: OpesAnalysisConfig) -> float:
    return config.energyunit_kj_per_mol / (8.314462618e-3 * config.temperature_k)


def log_mean_exp(values: np.ndarray, axis: int) -> np.ndarray:
    max_values = np.nanmax(values, axis=axis, keepdims=True)
    shifted = np.exp(values - max_values)
    return np.squeeze(max_values, axis=axis) + np.log(np.nanmean(shifted, axis=axis))


def load_flooding_set(barrier_dir: Path, barrier_value: float, config: OpesAnalysisConfig) -> dict[str, object]:
    colvars, log_files = collect_completed_run_files(
        barrier_dir,
        config.run_dir_prefix,
        config.colvar_glob,
        config.log_glob,
    )
    data = RM.get_data(colvars, 0, config.bias_col, acc_col=None, time_scale_factor=config.timeunit_seconds)
    event = RM.get_event(data, log_files=log_files, quiet=True)
    final_times = np.array([traj[-1, 0] for traj in data], dtype=float)
    obs_rate = float(event.sum() / final_times.sum())

    row_count = max(len(traj[:, 0]) for traj in data)
    v_data = np.full((len(data), row_count), np.nan)
    for traj_index, traj in enumerate(data):
        v_data[traj_index, : len(traj)] = traj[:, 1] + barrier_value

    ks_stat, p_value = ksc.ks_1samp_censored(final_times, event, lambda t: np.exp(-obs_rate * t))
    avg_bias_gamma1 = float(log_mean_exp(log_mean_exp(beta_value(config) * v_data, axis=0), axis=0))
    return {
        "label": barrier_dir.name,
        "barrier": barrier_value,
        "event": event,
        "final_times": final_times,
        "obs_rate": obs_rate,
        "log_obs_rate": float(np.log(obs_rate)),
        "v_data": v_data,
        "ks_stat": float(ks_stat),
        "p_value": float(p_value),
        "avg_bias_gamma1": avg_bias_gamma1,
        "avg_acceleration_factor_gamma1": float(np.exp(avg_bias_gamma1)),
        "run_count": len(colvars),
        "transitioned": int(event.sum()),
    }


def save_flooding_diagnostics_plot(
    title: str,
    diagnostics: dict[str, object],
    set_labels: list[str],
    output_path: Path,
    bootstrap_stats: dict[str, object] | None = None,
) -> None:
    plt = pyplot()
    gamma_grid = np.array(diagnostics["gamma_grid"], dtype=float)
    per_set = np.array(diagnostics["per_set_ln_k0"], dtype=float)
    mean_ln_k0 = np.array(diagnostics["mean_ln_k0"], dtype=float)
    var_ln_k0 = np.array(diagnostics["var_ln_k0"], dtype=float)
    gamma_best = float(diagnostics["gamma_best"])
    logk0_best = float(diagnostics["logk0_best"])

    fig, axes = plt.subplots(3, 1, figsize=(3.35, 6.85), sharex=True, gridspec_kw={"hspace": 0.04})

    for idx, label in enumerate(set_labels):
        axes[0].plot(gamma_grid, per_set[:, idx], label=label, color=SET_COLORS[idx % len(SET_COLORS)])
    axes[0].set_ylabel(r"Predicted ln($k_0$ / s$^{-1}$)")
    axes[0].legend(loc="lower left", ncol=2, handlelength=1.4, columnspacing=0.8)

    std_ln_k0 = np.sqrt(var_ln_k0)
    axes[1].plot(gamma_grid, mean_ln_k0, color=BLUE)
    axes[1].fill_between(gamma_grid, mean_ln_k0 - std_ln_k0, mean_ln_k0 + std_ln_k0, color=LIGHT_BLUE, alpha=0.9)
    axes[1].axvline(gamma_best, color=BLACK, linestyle="--", label=fr"min-var. $\gamma$ = {gamma_best:.2f}")
    axes[1].axhline(logk0_best, color=BLUE, linestyle="--", label=fr"mean ln($k_0$) = {logk0_best:.2f}")
    axes[1].set_ylabel(r"Mean ln($k_0$ / s$^{-1}$)")
    axes[1].legend(loc="lower left", handlelength=1.5)

    axes[2].plot(gamma_grid, var_ln_k0, color=BLACK)
    axes[2].axvline(gamma_best, color=BLACK, linestyle="--")
    axes[2].set_xlabel("gamma")
    axes[2].set_ylabel(r"Var[ln($k_0$)]")

    if bootstrap_stats is not None:
        fig.suptitle(
            f"{title}    bootstrap sigma(gamma*) = {float(bootstrap_stats['gamma_std']):.3f}, "
            f"sigma(ln k0*) = {float(bootstrap_stats['logk0_std']):.3f}",
            fontsize=10.0,
            y=0.985,
        )
    style_axes(axes)
    add_panel_labels(axes)
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    fig.subplots_adjust(top=0.93, bottom=0.08, left=0.22, right=0.98, hspace=0.04)

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_observed_ln_rate_vs_barrier(set_specs: list[dict[str, object]], bootstrap_stats: dict[str, object], output_path: Path) -> None:
    plt = pyplot()
    barriers = np.array([spec["barrier"] for spec in set_specs], dtype=float)
    ln_obs_rate = np.array([spec["log_obs_rate"] for spec in set_specs], dtype=float)
    ln_obs_rate_err = np.array(
        [entry["obs_rate_std"] / spec["obs_rate"] for spec, entry in zip(set_specs, bootstrap_stats["per_set"])],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(3.35, 2.23), constrained_layout=True)
    ax.errorbar(
        barriers,
        ln_obs_rate,
        yerr=ln_obs_rate_err,
        marker="o",
        capsize=2.5,
        color=BLUE,
        ecolor=BLUE,
        elinewidth=1.0,
        markerfacecolor=BLUE,
        markeredgecolor=BLUE,
    )
    for spec, xval, yval in zip(set_specs, barriers, ln_obs_rate):
        ax.annotate(str(spec["barrier"]), (xval, yval), textcoords="offset points", xytext=(4, 4), fontsize=8, color=GRAY)
    ax.set_xlabel(r"OPES barrier (kcal mol$^{-1}$)")
    ax.set_ylabel(r"Observed ln($k_{\mathrm{obs}}$ / s$^{-1}$)")
    style_axis(ax)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_ln_kobs_vs_acceleration(
    set_specs: list[dict[str, object]],
    diagnostics: dict[str, object],
    bootstrap_stats: dict[str, object],
    output_path: Path,
    title: str,
    config: OpesAnalysisConfig,
) -> None:
    plt = pyplot()
    ln_acceleration = np.array([spec["avg_bias_gamma1"] for spec in set_specs], dtype=float)
    ln_kobs = np.array([spec["log_obs_rate"] for spec in set_specs], dtype=float)
    ln_kobs_err = np.array(
        [entry["obs_rate_std"] / spec["obs_rate"] for spec, entry in zip(set_specs, bootstrap_stats["per_set"])],
        dtype=float,
    )
    slope, intercept = np.polyfit(ln_acceleration, ln_kobs, 1)
    x_fit = np.linspace(
        float(np.min(ln_acceleration)) * 0.98,
        float(np.max(ln_acceleration)) * 1.02,
        200,
    )
    y_fit = intercept + slope * x_fit

    fig, ax = plt.subplots(figsize=(3.35, 2.23), constrained_layout=True)
    ax.errorbar(
        ln_acceleration,
        ln_kobs,
        yerr=ln_kobs_err,
        marker="o",
        linestyle="none",
        capsize=2.5,
        color=BLUE,
        ecolor=BLUE,
        elinewidth=1.0,
        markerfacecolor=BLUE,
        markeredgecolor=BLUE,
        label="barrier sets",
    )
    ax.plot(x_fit, y_fit, color=BLACK, label="linear fit")
    for spec, x_value, y_value in zip(set_specs, ln_acceleration, ln_kobs):
        ax.annotate(str(spec["barrier"]), (x_value, y_value), textcoords="offset points", xytext=(4, 4), fontsize=8, color=GRAY)
    ax.set_xlabel(r"ln acceleration factor, ln($\alpha$)")
    ax.set_ylabel(r"ln($k_{\mathrm{obs}}$ / s$^{-1}$)")
    ax.text(
        0.03,
        0.97,
        f"linear-fit slope = {slope:.3f}\nlinear-fit intercept = {intercept:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": GRAY, "linewidth": 0.6},
    )
    style_axis(ax)
    ax.legend(loc="lower right", handlelength=1.5)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def analyze_cv(cv_dir: Path, output_root: Path, threads: int, numboots: int, config: OpesAnalysisConfig) -> Path:
    cv_output_dir = output_root / cv_dir.name
    cv_output_dir.mkdir(parents=True, exist_ok=True)

    set_specs = []
    for barrier_dir in sorted_prefixed_dirs(cv_dir, config.barrier_dir_prefix):
        barrier_value = float(barrier_dir.name.replace(config.barrier_dir_prefix, ""))
        if barrier_value > config.max_barrier_included:
            continue
        set_specs.append(load_flooding_set(barrier_dir, barrier_value, config))

    if not set_specs:
        raise SystemExit(f"No barrier directories <= {config.max_barrier_included:g} found under {cv_dir}")

    diagnostics = flooding_diagnostics(set_specs, threads=threads)
    rng = np.random.default_rng(20260504)
    bootstrap_stats = bootstrap_flooding(set_specs, numboots, rng, threads=threads)

    payload = {
        "cv": cv_dir.name,
        "config_path": str(config.config_path),
        "temperature_K": config.temperature_k,
        "energyunit_kj_per_mol": config.energyunit_kj_per_mol,
        "timeunit_seconds": config.timeunit_seconds,
        "bootstrap_resamples": numboots,
        "max_barrier_included": config.max_barrier_included,
        "sets": [
            {
                "set": spec["label"],
                "barrier": spec["barrier"],
                "transitioned": spec["transitioned"],
                "run_count": spec["run_count"],
                "obs_rate": spec["obs_rate"],
                "log_obs_rate": spec["log_obs_rate"],
                "ks_stat": spec["ks_stat"],
                "p_value": spec["p_value"],
                "ln_avg_exp_beta_v_gamma1": spec["avg_bias_gamma1"],
                "avg_acceleration_factor_gamma1": spec["avg_acceleration_factor_gamma1"],
            }
            for spec in set_specs
        ],
        "flooding_fit": diagnostics,
        "flooding_fit_bootstrap": bootstrap_stats,
    }
    summary_path = cv_output_dir / f"{cv_dir.name}_flooding_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    save_flooding_diagnostics_plot(
        f"{cv_dir.name} OPES EATR-flooding",
        diagnostics,
        [str(spec["barrier"]) for spec in set_specs],
        cv_output_dir / f"{cv_dir.name}_flooding_diagnostics.png",
        bootstrap_stats=bootstrap_stats,
    )
    plot_observed_ln_rate_vs_barrier(set_specs, bootstrap_stats, cv_output_dir / f"{cv_dir.name}_ln_kobs_vs_barrier.png")
    plot_ln_kobs_vs_acceleration(
        set_specs,
        diagnostics,
        bootstrap_stats,
        cv_output_dir / f"{cv_dir.name}_ln_kobs_vs_acceleration.png",
        f"{cv_dir.name} OPES slope-style rate scaling",
        config,
    )
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="dataset config TOML")
    parser.add_argument("--input-root", type=Path, default=None, help="override input directory containing OPES CV folders")
    parser.add_argument("--output-root", type=Path, default=None, help="override output directory for analysis products")
    parser.add_argument("--threads", type=int, default=None, help="number of threads for flooding diagnostics/bootstrap")
    parser.add_argument("--numboots", type=int, default=None, help="number of bootstrap replicas")
    parser.add_argument("--cv", nargs="+", default=None, help="optional subset of CV folder names to analyze")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_plot_env()
    config = load_opes_config(args.config)

    input_root = args.input_root.resolve() if args.input_root is not None else config.input_root
    output_root = args.output_root.resolve() if args.output_root is not None else config.output_root
    threads = args.threads if args.threads is not None else config.default_threads
    numboots = args.numboots if args.numboots is not None else config.bootstrap_resamples
    output_root.mkdir(parents=True, exist_ok=True)

    cv_dirs = sorted(
        [path for path in input_root.iterdir() if path.is_dir() and path.name.startswith(config.cv_dir_prefix)]
    )
    if args.cv is not None:
        wanted = set(args.cv)
        cv_dirs = [path for path in cv_dirs if path.name in wanted]
    if not cv_dirs:
        raise SystemExit(f"No CV directories found under {input_root}")

    summary_paths = thread_map(lambda cv_dir: analyze_cv(cv_dir, output_root, threads, numboots, config), cv_dirs, 1)
    manifest = {
        "config_path": str(config.config_path),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "threads": threads,
        "bootstrap_resamples": numboots,
        "summaries": [str(path) for path in summary_paths],
    }
    with open(output_root / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
