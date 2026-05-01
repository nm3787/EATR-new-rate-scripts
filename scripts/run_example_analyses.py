from __future__ import annotations

import json
import math
import os
from argparse import Namespace
from pathlib import Path

import numpy as np
from scipy import optimize

import ks_censored as ksc
import rate_methods_library as RM
ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = ROOT / "example-data" / "Ree_Data"
OUTPUT_ROOT = ROOT / "example-data" / "test_results"
MPL_CONFIG_DIR = ROOT / ".matplotlib-cache"
XDG_CACHE_DIR = ROOT / ".cache"

TIMEUNIT_SECONDS = 1e-15
TEMPERATURE_K = 312.0


def ensure_output_root() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
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


def sorted_run_files(base_dir: Path, filename: str) -> list[str]:
    return [str(path) for path in sorted(base_dir.glob(f"run_*/{filename}"))]


def pace_to_ps(pace_steps: float) -> float:
    return pace_steps * 10.0 * 1e-3


def run_regular_wt_eatr() -> dict[str, object]:
    wt_root = EXAMPLE_ROOT / "E_end_end_distance_wt"
    pace_dirs = sorted(wt_root.glob("eruns_pace*"), key=lambda path: float(path.name.split("pace")[1]))
    summaries: list[dict[str, float | str]] = []
    beta = beta_value()

    for pace_dir in pace_dirs:
        pace_steps = float(pace_dir.name.split("pace")[1])
        data = RM.get_data(sorted_run_files(pace_dir, "metad.colvar"), 0, 2, acc_col=4, time_scale_factor=TIMEUNIT_SECONDS)
        event = RM.get_event(data, log_files=sorted_run_files(pace_dir, "p.log"), quiet=True)
        mle_rate, mle_gamma = RM.EATR_MLE_rate(data, beta, event=event, gamma_bounds=(0.0, 1.0), cores=1, logTrick=False, do_bopt=False, bias_shift=0.0)
        log_average_exp_mle = RM.avg_exponential(data, beta, mle_gamma, bias_shift=0.0)
        final_time_indices = np.array([int(len(traj) - 1) for traj in data], dtype=int)
        mle_ks_stat, mle_p_value = ks_censored_ks(final_time_indices[event], np.exp(np.log(mle_rate)), log_average_exp_mle)

        cdf_ln_k = math.nan
        cdf_gamma = math.nan
        cdf_ks_stat = math.nan
        cdf_p_value = math.nan
        cdf_status = "ok"
        try:
            cdf_rate, cdf_gamma_value = RM.EATR_CDF_rate(data, beta, event=event, k_bounds=(1e-30, np.inf), gamma_bounds=(0.0, 1.0), cores=1, logTrick=False, do_bopt=False, bias_shift=0.0)
            cdf_ln_k = float(np.log(cdf_rate))
            cdf_gamma = float(cdf_gamma_value)
            log_average_exp_cdf = RM.avg_exponential(data, beta, cdf_gamma_value, bias_shift=0.0)
            cdf_ks_stat, cdf_p_value = ks_censored_ks(final_time_indices[event], cdf_rate, log_average_exp_cdf)
        except Exception as exc:
            cdf_status = f"failed: {type(exc).__name__}"

        summaries.append(
            {
                "set": pace_dir.name,
                "pace_steps": pace_steps,
                "pace_ps": pace_to_ps(pace_steps),
                "transitioned": int(event.sum()),
                "total": len(event),
                "eatr_mle_ln_k": float(np.log(mle_rate)),
                "eatr_mle_gamma": float(mle_gamma),
                "eatr_cdf_ln_k": cdf_ln_k,
                "eatr_cdf_gamma": cdf_gamma,
                "eatr_mle_ks_stat": float(mle_ks_stat),
                "eatr_mle_p_value": float(mle_p_value),
                "eatr_cdf_ks_stat": cdf_ks_stat,
                "eatr_cdf_p_value": cdf_p_value,
                "eatr_cdf_status": cdf_status,
            }
        )

    output = {"temperature_K": TEMPERATURE_K, "timeunit_seconds": TIMEUNIT_SECONDS, "sets": summaries}
    with open(OUTPUT_ROOT / "wt_regular_eatr_summary.json", "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    plot_regular_eatr(summaries)
    return output


def plot_regular_eatr(summaries: list[dict[str, float | str]]) -> None:
    plt = pyplot()
    pace_ps = np.array([entry["pace_ps"] for entry in summaries], dtype=float)
    ln_k_mle = np.array([entry["eatr_mle_ln_k"] for entry in summaries], dtype=float)
    ln_k_cdf = np.array([entry["eatr_cdf_ln_k"] for entry in summaries], dtype=float)
    gamma_mle = np.array([entry["eatr_mle_gamma"] for entry in summaries], dtype=float)
    gamma_cdf = np.array([entry["eatr_cdf_gamma"] for entry in summaries], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(pace_ps, ln_k_mle, marker="o", label="EATR MLE")
    axes[0].plot(pace_ps, ln_k_cdf, marker="s", label="EATR CDF")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("MetaD hill deposition pace (ps)")
    axes[0].set_ylabel("Estimated ln k0")
    axes[0].set_title("WT-MetaD Regular EATR")
    axes[0].legend()

    axes[1].plot(pace_ps, gamma_mle, marker="o", label="EATR MLE")
    axes[1].plot(pace_ps, gamma_cdf, marker="s", label="EATR CDF")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("MetaD hill deposition pace (ps)")
    axes[1].set_ylabel("Estimated γ")
    axes[1].set_title("WT-MetaD Biasing Efficiency")
    axes[1].legend()

    fig.savefig(OUTPUT_ROOT / "wt_regular_eatr_vs_pace.png", dpi=200)
    plt.close(fig)


def ks_censored_ks(final_time_indices: np.ndarray, rate: float, log_average_exp: np.ndarray) -> tuple[float, float]:
    return ksc.ks_1samp_censored(
        final_time_indices,
        np.array([True for _ in final_time_indices]),
        lambda idx: np.exp(-rate * RM.EATR_calculate_cum_hazard(log_average_exp, False, idx)),
    )


def load_flooding_set(colvar_files: list[str], log_files: list[str], bias_col: int, acc_col: int | None, bias_shift: float, cdf_fit: bool) -> dict[str, object]:
    data = RM.get_data(colvar_files, 0, bias_col, acc_col=acc_col, time_scale_factor=TIMEUNIT_SECONDS)
    event = RM.get_event(data, log_files=log_files, quiet=True)
    final_times = np.array([traj[-1, 0] for traj in data], dtype=float)
    obs_rate = float(event.sum() / final_times.sum())
    if cdf_fit:
        ecdfxs = np.sort(final_times)
        ecdfys = np.linspace(1 / len(event), 1, len(event))
        obs_rate = float(optimize.curve_fit(lambda t, k: 1 - np.exp(-k * t), ecdfxs[: event.sum()], ecdfys[: event.sum()], p0=obs_rate)[0][0])
    ks_stat, p_value = ksc.ks_1samp_censored(final_times, event, lambda t: np.exp(-obs_rate * t))

    colvar_row_counts = np.sort([len(traj[:, 0]) for traj in data])
    max_index = colvar_row_counts[-1]
    min_index = 0
    v_data = np.full((len(data), max_index - min_index), np.nan)
    for traj_index, traj in enumerate(data):
        v_data[traj_index, : (min(len(traj), max_index) - min_index)] = traj[min_index:max_index, 1] + bias_shift

    return {
        "data": data,
        "event": event,
        "obs_rate": obs_rate,
        "ks_stat": float(ks_stat),
        "p_value": float(p_value),
        "v_data": v_data,
        "avg_bias_gamma1": float(np.log(np.mean(np.nanmean(np.exp(beta_value() * v_data), axis=0)))),
    }


def beta_value() -> float:
    return 1.0 / (8.314462618e-3 * TEMPERATURE_K)


def flooding_diagnostics(set_specs: list[dict[str, object]], axis_first: int = 0) -> dict[str, object]:
    beta = beta_value()
    gamma_grid = np.linspace(0.0, 1.0, 401)
    per_set_ln_k0: list[list[float]] = []
    mean_ln_k0 = []
    var_ln_k0 = []

    for gamma in gamma_grid:
        ln_k0s = []
        for spec in set_specs:
            avg = np.mean(np.nanmean(np.exp(beta * gamma * spec["v_data"]), axis=axis_first))
            ln_k0s.append(float(np.log(spec["obs_rate"]) - np.log(avg)))
        per_set_ln_k0.append(ln_k0s)
        mean_ln_k0.append(float(np.mean(ln_k0s)))
        var_ln_k0.append(float(np.var(ln_k0s)))

    objective = lambda gamma: np.var(
        [
            np.log(spec["obs_rate"]) - np.log(np.mean(np.nanmean(np.exp(beta * gamma * spec["v_data"]), axis=axis_first)))
            for spec in set_specs
        ]
    )
    optimum = optimize.minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")
    gamma_best = float(optimum.x)
    avg_terms = [float(np.mean(np.nanmean(np.exp(beta * gamma_best * spec["v_data"]), axis=axis_first))) for spec in set_specs]
    ln_k0_per_set_best = [float(np.log(spec["obs_rate"]) - np.log(avg_term)) for spec, avg_term in zip(set_specs, avg_terms)]
    logk0_best = float(np.mean(ln_k0_per_set_best))

    return {
        "gamma_grid": gamma_grid.tolist(),
        "per_set_ln_k0": per_set_ln_k0,
        "mean_ln_k0": mean_ln_k0,
        "var_ln_k0": var_ln_k0,
        "gamma_best": gamma_best,
        "logk0_best": logk0_best,
        "ln_k0_per_set_best": ln_k0_per_set_best,
    }


def save_flooding_plot(title: str, diagnostics: dict[str, object], set_labels: list[str], output_name: str, reference_lines: dict[str, float] | None = None) -> None:
    plt = pyplot()
    gamma_grid = np.array(diagnostics["gamma_grid"], dtype=float)
    per_set = np.array(diagnostics["per_set_ln_k0"], dtype=float)
    mean_ln_k0 = np.array(diagnostics["mean_ln_k0"], dtype=float)
    var_ln_k0 = np.array(diagnostics["var_ln_k0"], dtype=float)
    gamma_best = float(diagnostics["gamma_best"])
    logk0_best = float(diagnostics["logk0_best"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    for idx, label in enumerate(set_labels):
        axes[0].plot(gamma_grid, per_set[:, idx], label=label)
    axes[0].set_xlabel("γ")
    axes[0].set_ylabel("Predicted ln k0")
    axes[0].set_title(f"{title}: ln k0 by set")
    axes[0].legend(fontsize=8, ncol=2)

    std_ln_k0 = np.sqrt(var_ln_k0)
    axes[1].plot(gamma_grid, mean_ln_k0, color="black")
    axes[1].fill_between(gamma_grid, mean_ln_k0 - std_ln_k0, mean_ln_k0 + std_ln_k0, color="tab:blue", alpha=0.2)
    axes[1].axvline(gamma_best, color="tab:red", linestyle="--", label=f"γ*={gamma_best:.3f}")
    axes[1].axhline(logk0_best, color="tab:green", linestyle="--", label=f"ln k0*={logk0_best:.3f}")
    if reference_lines:
        for name, value in reference_lines.items():
            axes[1].axhline(value, linestyle=":", label=f"{name}={value:.3f}")
    axes[1].set_xlabel("γ")
    axes[1].set_ylabel("Mean ln k0")
    axes[1].set_title(f"{title}: mean ± std")
    axes[1].legend(fontsize=8)

    axes[2].plot(gamma_grid, var_ln_k0, color="tab:purple")
    axes[2].axvline(gamma_best, color="tab:red", linestyle="--")
    axes[2].set_xlabel("γ")
    axes[2].set_ylabel("Var[ln k0]")
    axes[2].set_title(f"{title}: variance")

    fig.savefig(OUTPUT_ROOT / output_name, dpi=220)
    plt.close(fig)


def run_opes_flooding() -> dict[str, object]:
    opes_root = EXAMPLE_ROOT / "E_end_end_distance_opes"
    set_specs = []
    labels = []
    barriers = []
    for path in sorted(opes_root.glob("eruns_barr*"), key=lambda item: float(item.name.split("barr")[1])):
        barrier = float(path.name.split("barr")[1])
        set_info = load_flooding_set(sorted_run_files(path, "opes_short.colvar"), sorted_run_files(path, "p.log"), bias_col=4, acc_col=None, bias_shift=barrier, cdf_fit=False)
        set_info["label"] = path.name
        set_info["barrier_kj_mol"] = barrier
        set_specs.append(set_info)
        labels.append(path.name)
        barriers.append(barrier)

    diagnostics = flooding_diagnostics(set_specs)
    output = {
        "temperature_K": TEMPERATURE_K,
        "timeunit_seconds": TIMEUNIT_SECONDS,
        "sets": [
            {
                "set": spec["label"],
                "barrier_kj_mol": spec["barrier_kj_mol"],
                "obs_rate": spec["obs_rate"],
                "ks_stat": spec["ks_stat"],
                "p_value": spec["p_value"],
                "ln_avg_exp_beta_v_gamma1": spec["avg_bias_gamma1"],
            }
            for spec in set_specs
        ],
        "flooding_fit": diagnostics,
    }
    with open(OUTPUT_ROOT / "opes_flooding_summary.json", "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    save_flooding_plot("OPES EATR-flooding", diagnostics, labels, "opes_flooding_diagnostics.png")
    return output


def run_wt_flooding() -> dict[str, object]:
    wt_root = EXAMPLE_ROOT / "E_end_end_distance_wt"
    pace_dirs = sorted(wt_root.glob("eruns_pace*"), key=lambda path: float(path.name.split("pace")[1]))
    set_specs = []
    labels = []
    pace_ps_values = []
    for pace_dir in pace_dirs:
        pace_steps = float(pace_dir.name.split("pace")[1])
        set_info = load_flooding_set(sorted_run_files(pace_dir, "metad.colvar"), sorted_run_files(pace_dir, "p.log"), bias_col=2, acc_col=4, bias_shift=0.0, cdf_fit=False)
        set_info["label"] = pace_dir.name
        set_info["pace_steps"] = pace_steps
        set_info["pace_ps"] = pace_to_ps(pace_steps)
        set_specs.append(set_info)
        labels.append(pace_dir.name)
        pace_ps_values.append(set_info["pace_ps"])

    diagnostics_all = flooding_diagnostics(set_specs)
    filtered_specs = [spec for spec in set_specs if float(spec["pace_ps"]) >= 100.0]
    filtered_labels = [spec["label"] for spec in filtered_specs]
    diagnostics_filtered = flooding_diagnostics(filtered_specs)

    output = {
        "temperature_K": TEMPERATURE_K,
        "timeunit_seconds": TIMEUNIT_SECONDS,
        "sets": [
            {
                "set": spec["label"],
                "pace_steps": spec["pace_steps"],
                "pace_ps": spec["pace_ps"],
                "obs_rate": spec["obs_rate"],
                "ks_stat": spec["ks_stat"],
                "p_value": spec["p_value"],
                "ln_avg_exp_beta_v_gamma1": spec["avg_bias_gamma1"],
            }
            for spec in set_specs
        ],
        "flooding_fit_all_sets": diagnostics_all,
        "flooding_fit_filtered_pace_ge_100ps": diagnostics_filtered,
    }
    with open(OUTPUT_ROOT / "wt_flooding_summary.json", "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    save_flooding_plot("WT-MetaD EATR-flooding (all paces)", diagnostics_all, labels, "wt_flooding_all_paces.png")
    save_flooding_plot("WT-MetaD EATR-flooding (pace ≥ 100 ps)", diagnostics_filtered, filtered_labels, "wt_flooding_filtered_paces.png")
    plot_wt_observed_rate_vs_pace(set_specs, diagnostics_filtered)
    return output


def plot_wt_observed_rate_vs_pace(set_specs: list[dict[str, object]], diagnostics_filtered: dict[str, object]) -> None:
    plt = pyplot()
    pace_ps = np.array([spec["pace_ps"] for spec in set_specs], dtype=float)
    obs_ln_k = np.log(np.array([spec["obs_rate"] for spec in set_specs], dtype=float))
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    ax.plot(pace_ps, obs_ln_k, marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("MetaD hill deposition pace (ps)")
    ax.set_ylabel("Observed ln k_obs")
    ax.set_title("WT-MetaD observed rate by pace")
    ax.axhline(float(diagnostics_filtered["logk0_best"]), color="tab:red", linestyle="--", label="Filtered flooding ln k0*")
    ax.legend()
    fig.savefig(OUTPUT_ROOT / "wt_observed_rate_vs_pace.png", dpi=220)
    plt.close(fig)


def main() -> None:
    ensure_output_root()
    regular = run_regular_wt_eatr()
    opes = run_opes_flooding()
    wt_flooding = run_wt_flooding()

    manifest = {
        "generated_files": sorted(path.name for path in OUTPUT_ROOT.iterdir()),
        "notes": [
            "Protein G example trajectories use a 10 fs timestep in LAMMPS real units, so times were converted with 1e-15 s/fs.",
            "WT flooding analysis is reported for all pace sets and for a manuscript-style filtered subset with pace >= 100 ps.",
        ],
        "wt_regular_summary": "wt_regular_eatr_summary.json",
        "opes_flooding_summary": "opes_flooding_summary.json",
        "wt_flooding_summary": "wt_flooding_summary.json",
    }
    with open(OUTPUT_ROOT / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()
