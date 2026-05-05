from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eatr_rates.dataset_config import ImetadAnalysisConfig, load_imetad_config
from eatr_rates.analysis_io import collect_completed_run_files, sorted_prefixed_dirs
from scripts.run_example_analyses import (
    bootstrap_regular_eatr,
    build_prepared_data,
    eatr_mle_from_prepared,
    ks_censored_ks,
    thread_map,
)
import rate_methods_library as RM

DEFAULT_CONFIG_PATH = ROOT / "analysis-configs" / "nnp_imetad.toml"
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

    return plt


def beta_value(config: ImetadAnalysisConfig) -> float:
    return config.energyunit_kj_per_mol / (8.314462618e-3 * config.temperature_k)


def pace_to_ps(pace_steps: float, config: ImetadAnalysisConfig) -> float:
    return pace_steps * config.timestep_ps


def load_regular_set(pace_dir: Path, beta: float, numboots: int, threads: int, rng: np.random.Generator, config: ImetadAnalysisConfig) -> dict[str, object]:
    colvars, log_files = collect_completed_run_files(
        pace_dir,
        config.run_dir_prefix,
        config.colvar_glob,
        config.log_glob,
    )

    data = RM.get_data(colvars, 0, config.bias_col, acc_col=config.acc_col, time_scale_factor=config.timeunit_seconds)
    event = RM.get_event(data, log_files=log_files, quiet=True)
    prepared = build_prepared_data(data, event, bias_shift=0.0)
    final_time_indices = np.asarray(prepared["final_time_indices"], dtype=int)

    mle_fit = eatr_mle_from_prepared(prepared, beta)
    mle_rate = float(mle_fit["k0"])
    mle_gamma = float(mle_fit["gamma"])
    mle_log_average_exp = np.asarray(mle_fit["log_average_exp"], dtype=float)
    mle_ks_stat, mle_p_value = ks_censored_ks(final_time_indices[event], mle_rate, mle_log_average_exp)
    mle_bootstrap = bootstrap_regular_eatr(prepared, beta, numboots, rng, threads=threads)

    cdf_ln_k = math.nan
    cdf_gamma = math.nan
    cdf_ks_stat = math.nan
    cdf_p_value = math.nan
    cdf_status = "ok"
    try:
        cdf_rate, cdf_gamma_value = RM.EATR_CDF_rate(
            data,
            beta,
            event=event,
            k_bounds=(1e-30, np.inf),
            gamma_bounds=(0.0, 1.0),
            cores=1,
            logTrick=False,
            do_bopt=False,
            bias_shift=0.0,
        )
        cdf_ln_k = float(np.log(cdf_rate))
        cdf_gamma = float(cdf_gamma_value)
        cdf_log_average_exp = RM.avg_exponential(data, beta, cdf_gamma_value, bias_shift=0.0)
        cdf_ks_stat, cdf_p_value = ks_censored_ks(final_time_indices[event], cdf_rate, cdf_log_average_exp)
    except Exception as exc:
        cdf_status = f"failed: {type(exc).__name__}"

    pace_steps = float(pace_dir.name.replace(config.pace_dir_prefix, ""))
    return {
        "set": pace_dir.name,
        "pace_steps": pace_steps,
        "pace_ps": pace_to_ps(pace_steps, config),
        "transitioned": int(event.sum()),
        "total": len(event),
        "completed_runs": len(colvars),
        "eatr_mle_ln_k": float(np.log(mle_rate)),
        "eatr_mle_gamma": mle_gamma,
        "eatr_cdf_ln_k": cdf_ln_k,
        "eatr_cdf_gamma": cdf_gamma,
        "eatr_mle_ks_stat": float(mle_ks_stat),
        "eatr_mle_p_value": float(mle_p_value),
        "eatr_mle_bootstrap_n": int(mle_bootstrap["n_resamples"]),
        "eatr_mle_ln_k_std": float(mle_bootstrap["ln_k_std"]),
        "eatr_mle_ln_k_ci95_low": float(mle_bootstrap["ln_k_ci95_low"]),
        "eatr_mle_ln_k_ci95_high": float(mle_bootstrap["ln_k_ci95_high"]),
        "eatr_mle_gamma_std": float(mle_bootstrap["gamma_std"]),
        "eatr_mle_gamma_ci95_low": float(mle_bootstrap["gamma_ci95_low"]),
        "eatr_mle_gamma_ci95_high": float(mle_bootstrap["gamma_ci95_high"]),
        "eatr_cdf_ks_stat": float(cdf_ks_stat) if np.isfinite(cdf_ks_stat) else cdf_ks_stat,
        "eatr_cdf_p_value": float(cdf_p_value) if np.isfinite(cdf_p_value) else cdf_p_value,
        "eatr_cdf_status": cdf_status,
    }


def plot_regular_series(summaries: list[dict[str, object]], output_path: Path, title: str) -> None:
    plt = pyplot()
    pace_ps = np.array([entry["pace_ps"] for entry in summaries], dtype=float)
    ln_k_mle = np.array([entry["eatr_mle_ln_k"] for entry in summaries], dtype=float)
    ln_k_cdf = np.array([entry["eatr_cdf_ln_k"] for entry in summaries], dtype=float)
    gamma_mle = np.array([entry["eatr_mle_gamma"] for entry in summaries], dtype=float)
    gamma_cdf = np.array([entry["eatr_cdf_gamma"] for entry in summaries], dtype=float)
    ln_k_mle_err = np.array([entry["eatr_mle_ln_k_std"] for entry in summaries], dtype=float)
    gamma_mle_err = np.array([entry["eatr_mle_gamma_std"] for entry in summaries], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].errorbar(pace_ps, ln_k_mle, yerr=ln_k_mle_err, marker="o", capsize=3, label="EATR MLE bootstrap sigma")
    axes[0].plot(pace_ps, ln_k_cdf, marker="s", label="EATR CDF")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("MetaD hill deposition pace (ps)")
    axes[0].set_ylabel(r"Estimated ln($k_0$ / s$^{-1}$)")
    axes[0].set_title(title)
    axes[0].legend()

    axes[1].errorbar(pace_ps, gamma_mle, yerr=gamma_mle_err, marker="o", capsize=3, label="EATR MLE bootstrap sigma")
    axes[1].plot(pace_ps, gamma_cdf, marker="s", label="EATR CDF")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("MetaD hill deposition pace (ps)")
    axes[1].set_ylabel("Estimated gamma")
    axes[1].set_title(f"{title}: biasing efficiency")
    axes[1].legend()

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_observed_ln_rate_vs_pace(summaries: list[dict[str, object]], output_path: Path, title: str) -> None:
    plt = pyplot()
    pace_ps = np.array([entry["pace_ps"] for entry in summaries], dtype=float)
    ln_kobs = np.array(
        [float(np.log(entry["transitioned"] / entry["total"]) + entry["eatr_mle_ln_k"]) for entry in summaries],
        dtype=float,
    )
    ln_kobs_err = np.array([entry["eatr_mle_ln_k_std"] for entry in summaries], dtype=float)

    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    ax.errorbar(pace_ps, ln_kobs, yerr=ln_kobs_err, marker="o", capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("MetaD hill deposition pace (ps)")
    ax.set_ylabel(r"Observed ln($k_{\mathrm{obs}}$ / s$^{-1}$)")
    ax.set_title(f"{title}: observed rate")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def analyze_height(height_dir: Path, output_root: Path, threads: int, numboots: int, seed: int, config: ImetadAnalysisConfig) -> Path:
    beta = beta_value(config)
    height_output_dir = output_root / height_dir.parent.name / height_dir.name
    height_output_dir.mkdir(parents=True, exist_ok=True)

    pace_dirs = sorted_prefixed_dirs(height_dir, config.pace_dir_prefix)
    rng = np.random.default_rng(seed)
    summaries = [load_regular_set(pace_dir, beta, numboots, threads, rng, config) for pace_dir in pace_dirs]

    payload = {
        "cv": height_dir.parent.name,
        "height": height_dir.name,
        "config_path": str(config.config_path),
        "temperature_K": config.temperature_k,
        "energyunit_kj_per_mol": config.energyunit_kj_per_mol,
        "timeunit_seconds": config.timeunit_seconds,
        "timestep_ps": config.timestep_ps,
        "bootstrap_resamples": numboots,
        "sets": summaries,
    }
    summary_path = height_output_dir / f"{height_dir.parent.name}_{height_dir.name}_regular_eatr_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    plot_regular_series(
        summaries,
        height_output_dir / f"{height_dir.parent.name}_{height_dir.name}_regular_eatr_vs_pace.png",
        f"{height_dir.parent.name} {height_dir.name} regular EATR",
    )
    plot_observed_ln_rate_vs_pace(
        summaries,
        height_output_dir / f"{height_dir.parent.name}_{height_dir.name}_ln_kobs_vs_pace.png",
        f"{height_dir.parent.name} {height_dir.name} regular EATR",
    )
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="dataset config TOML")
    parser.add_argument("--input-root", type=Path, default=None, help="override input directory containing iMetaD CV folders")
    parser.add_argument("--output-root", type=Path, default=None, help="override output directory for analysis products")
    parser.add_argument("--threads", type=int, default=None, help="number of threads for bootstrap work")
    parser.add_argument("--numboots", type=int, default=None, help="number of bootstrap replicas")
    parser.add_argument("--cv", nargs="+", default=None, help="optional subset of CV folder names to analyze")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_plot_env()
    config = load_imetad_config(args.config)

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

    height_dirs: list[Path] = []
    for cv_dir in cv_dirs:
        height_dirs.extend(sorted([path for path in cv_dir.iterdir() if path.is_dir() and path.name.startswith(config.height_dir_prefix)]))
    if not height_dirs:
        raise SystemExit(f"No height directories found under {input_root}")

    def worker(item: tuple[int, Path]) -> Path:
        index, height_dir = item
        return analyze_height(height_dir, output_root, threads, numboots, 20260504 + index, config)

    summary_paths = thread_map(worker, list(enumerate(height_dirs)), 1)
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
