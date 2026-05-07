from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
from eatr_rates.plot_style import (
    BLACK,
    BLUE,
    GRAY,
    LIGHT_BLUE,
    ORANGE,
    add_panel_labels,
    apply_publication_style,
    style_axis,
    style_axes,
)


def pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    apply_publication_style(plt)

    return plt


METHOD_KEYS = {
    "eatr-comparison": None,
    "imetad-mle": ("iMetaD MLE ln k", None),
    "imetad-cdf": ("iMetaD CDF ln k", None),
    "ktr-mle": ("KTR MLE ln k", "KTR MLE gamma"),
    "ktr-cdf": ("KTR CDF ln k", "KTR CDF gamma"),
    "eatr-mle": ("EATR MLE ln k", "EATR MLE gamma"),
    "eatr-cdf": ("EATR CDF ln k", "EATR CDF gamma"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    regular = subparsers.add_parser("regular-series", help="plot a regular-analysis series from multiple JSON outputs")
    regular.add_argument("-i", "--input", nargs="+", required=True, help="JSON outputs from eatr-analysis")
    regular.add_argument("--xvalues", nargs="+", type=float, default=None, help="x-axis values corresponding to the input JSON files; if omitted, infer pace values from filenames like pace_100ps.json")
    regular.add_argument("--labels", nargs="+", default=None, help="optional point labels matching the input JSON files")
    regular.add_argument("--xlabel", type=str, default="Condition", help="x-axis label")
    regular.add_argument("--xscale", choices=["linear", "log"], default="log", help="x-axis scaling")
    regular.add_argument("--method", choices=sorted(METHOD_KEYS), default="eatr-mle", help="which method keys to plot")
    regular.add_argument("--noline", action="store_true", help="remove the connecting lines in the plots")
    regular.add_argument("--truerate", type=np.float64, default=None, help="optional true rate to compare to results")
    regular.add_argument("-o", "--output", type=str, default="regular_series.png", help="output figure path")

    flooding = subparsers.add_parser("flooding", help="plot figures from one eatr-flooding-analysis JSON output")
    flooding.add_argument("-i", "--input", required=True, help="JSON output from eatr-flooding-analysis")
    flooding.add_argument("--condition-label", type=str, default="Bias label", help="label for the per-set condition values")
    flooding.add_argument("--condition-unit", type=str, default="", help="unit suffix for the per-set condition values")
    flooding.add_argument("--title-prefix", type=str, default="Flooding analysis", help="title prefix for the generated figures")
    flooding.add_argument("-o", "--output-prefix", type=str, default="flooding", help="prefix for generated figure files")

    return parser


def load_json(path: str) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_uncertainty(payload: dict[str, object], key: str):
    std_key = f"{key} std"
    if std_key in payload:
        std = float(payload[std_key])
        return np.array([std]), np.array([std])
    ci_key = f"{key} CI"
    if ci_key in payload:
        low, high = payload[ci_key]
        value = float(payload[key])
        return np.array([value - float(low)]), np.array([float(high) - value])
    return None


PACE_PATTERN = re.compile(r"pace[_-]?([0-9]+(?:\.[0-9]+)?)ps$", re.IGNORECASE)


def autodetect_xvalues(paths: list[str]) -> np.ndarray:
    values = []
    for path in paths:
        match = PACE_PATTERN.search(Path(path).stem)
        if match is None:
            raise SystemExit(
                "Could not infer x values from input filenames. "
                "Use --xvalues explicitly or name files like pace_100ps.json."
            )
        values.append(float(match.group(1)))
    return np.array(values, dtype=float)


def apply_xlimits(axis, xvalues: np.ndarray, xscale: str) -> None:
    xmin = float(np.min(xvalues))
    xmax = float(np.max(xvalues))
    if xscale == "log":
        if xmin <= 0.0:
            raise SystemExit("Log-scaled plots require strictly positive x values.")
        axis.set_xlim(xmin / 1.5, xmax * 1.5)
        return
    span = xmax - xmin
    pad = 0.05 * span if span > 0.0 else max(0.05 * abs(xmin), 0.5)
    axis.set_xlim(xmin - pad, xmax + pad)


def plot_regular_series(args: argparse.Namespace) -> int:
    if args.xvalues is not None and len(args.input) != len(args.xvalues):
        raise SystemExit("The number of --input files must match the number of --xvalues.")
    if args.labels is not None and len(args.labels) != len(args.input):
        raise SystemExit("If provided, --labels must match the number of --input files.")

    payloads = [load_json(path) for path in args.input]
    xvalues = autodetect_xvalues(args.input) if args.xvalues is None else np.array(args.xvalues, dtype=float)
    labels = args.labels if args.labels is not None else [Path(path).stem for path in args.input]
    linestyle = '' if args.noline else '-'

    if args.method == "eatr-comparison":
        return plot_eatr_comparison(payloads, xvalues, labels, args.xlabel, args.xscale, args.output)

    log_key, gamma_key = METHOD_KEYS[args.method]

    log_values = np.array([float(payload[log_key]) for payload in payloads], dtype=float)
    log_error = None
    if any(f"{log_key} std" in payload or f"{log_key} CI" in payload for payload in payloads):
        lower = []
        upper = []
        for payload in payloads:
            err = extract_uncertainty(payload, log_key)
            if err is None:
                lower.append(0.0)
                upper.append(0.0)
            else:
                lower.append(float(err[0][0]))
                upper.append(float(err[1][0]))
        log_error = np.array([lower, upper], dtype=float)

    gamma_values = None
    gamma_error = None
    if gamma_key is not None and all(gamma_key in payload for payload in payloads):
        gamma_values = np.array([float(payload[gamma_key]) for payload in payloads], dtype=float)
        if any(f"{gamma_key} std" in payload or f"{gamma_key} CI" in payload for payload in payloads):
            lower = []
            upper = []
            for payload in payloads:
                err = extract_uncertainty(payload, gamma_key)
                if err is None:
                    lower.append(0.0)
                    upper.append(0.0)
                else:
                    lower.append(float(err[0][0]))
                    upper.append(float(err[1][0]))
            gamma_error = np.array([lower, upper], dtype=float)

    plt = pyplot()
    ncols = 2 if gamma_values is not None else 1
    if ncols == 2:
        fig, axes = plt.subplots(2, 1, figsize=(3.35, 5.35), sharex=True, gridspec_kw={"hspace": 0.04})
    else:
        fig, axes = plt.subplots(1, 1, figsize=(3.35, 2.23), constrained_layout=True)
        axes = [axes]

    if log_error is None:
        axes[0].plot(xvalues, log_values, linestyle=linestyle, marker="o", color=BLUE)
    else:
        axes[0].errorbar(
            xvalues, log_values, yerr=log_error, linestyle=linestyle, marker="o", capsize=2.5,
            color=BLUE, ecolor=BLUE, elinewidth=1.0, markerfacecolor=BLUE, markeredgecolor=BLUE
        )
    for label, xval, yval in zip(labels, xvalues, log_values):
        axes[0].annotate(label, (xval, yval), textcoords="offset points", xytext=(4, 4), fontsize=8, color=GRAY)
    axes[0].set_xscale(args.xscale)
    if args.truerate is not None:
        axes[0].axhline(args.truerate, linestyle='--', color=BLUE)
    apply_xlimits(axes[0], xvalues, args.xscale)
    axes[0].set_ylabel(r"Estimated ln($k_0$ / s$^{-1}$)")
    if ncols == 1:
        axes[0].set_xlabel(args.xlabel)
        style_axis(axes[0])

    if gamma_values is not None:
        axes[1].set_ylim((-0.05,1.05))
        axes[1].axhline(0.0, linestyle='--', color=BLACK)
        axes[1].axhline(1.0, linestyle='--', color=BLACK)
        if gamma_error is None:
            axes[1].plot(xvalues, gamma_values, linestyle=linestyle, marker="o", color=BLUE)
        else:
            axes[1].errorbar(
                xvalues, gamma_values, yerr=gamma_error, linestyle=linestyle, marker="o", capsize=2.5,
                color=BLUE, ecolor=BLUE, elinewidth=1.0, markerfacecolor=BLUE, markeredgecolor=BLUE
            )
        for label, xval, yval in zip(labels, xvalues, gamma_values):
            axes[1].annotate(label, (xval, yval), textcoords="offset points", xytext=(4, 4), fontsize=8, color=GRAY)
        axes[1].set_xscale(args.xscale)
        apply_xlimits(axes[1], xvalues, args.xscale)
        axes[1].set_xlabel(args.xlabel)
        axes[1].set_ylabel("Estimated γ")
        style_axes(axes)
        add_panel_labels(axes, ["(a)", "(b)"])
        axes[0].tick_params(labelbottom=False)
        fig.subplots_adjust(top=0.96, bottom=0.10, left=0.18, right=0.98, hspace=0.04)

    fig.savefig(args.output, dpi=220)
    plt.close(fig)
    return 0


def plot_eatr_comparison(payloads, xvalues, labels, xlabel: str, xscale: str, output: str) -> int:
    plt = pyplot()

    mle_ln_k = np.array([float(payload["EATR MLE ln k"]) for payload in payloads], dtype=float)
    mle_gamma = np.array([float(payload["EATR MLE gamma"]) for payload in payloads], dtype=float)
    cdf_ln_k = np.array([float(payload["EATR CDF ln k"]) if "EATR CDF ln k" in payload else np.nan for payload in payloads], dtype=float)
    cdf_gamma = np.array([float(payload["EATR CDF gamma"]) if "EATR CDF gamma" in payload else np.nan for payload in payloads], dtype=float)

    mle_ln_k_err = np.array([float(payload.get("EATR MLE ln k std", 0.0)) for payload in payloads], dtype=float)
    mle_gamma_err = np.array([float(payload.get("EATR MLE gamma std", 0.0)) for payload in payloads], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(3.35, 5.35), sharex=True, gridspec_kw={"hspace": 0.04})
    axes[0].errorbar(
        xvalues, mle_ln_k, yerr=mle_ln_k_err, marker="o", capsize=2.5, label="EATR MLE",
        color=BLUE, ecolor=BLUE, elinewidth=1.0, markerfacecolor=BLUE, markeredgecolor=BLUE
    )
    axes[0].plot(xvalues, cdf_ln_k, marker="s", color=ORANGE, label="EATR CDF")
    axes[0].set_xscale(xscale)
    apply_xlimits(axes[0], xvalues, xscale)
    axes[0].set_ylabel(r"Estimated ln($k_0$ / s$^{-1}$)")
    axes[0].legend(loc="best", handlelength=1.5)
    for label, xval, yval in zip(labels, xvalues, mle_ln_k):
        axes[0].annotate(label, (xval, yval), textcoords="offset points", xytext=(4, 4), fontsize=8, color=GRAY)

    axes[1].errorbar(
        xvalues, mle_gamma, yerr=mle_gamma_err, marker="o", capsize=2.5, label="EATR MLE",
        color=BLUE, ecolor=BLUE, elinewidth=1.0, markerfacecolor=BLUE, markeredgecolor=BLUE
    )
    axes[1].plot(xvalues, cdf_gamma, marker="s", color=ORANGE, label="EATR CDF")
    axes[1].set_xscale(xscale)
    apply_xlimits(axes[1], xvalues, xscale)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Estimated γ")
    axes[1].legend(loc="best", handlelength=1.5)
    for label, xval, yval in zip(labels, xvalues, mle_gamma):
        axes[1].annotate(label, (xval, yval), textcoords="offset points", xytext=(4, 4), fontsize=8, color=GRAY)
    style_axes(axes)
    add_panel_labels(axes, ["(a)", "(b)"])
    axes[0].tick_params(labelbottom=False)
    fig.subplots_adjust(top=0.98, bottom=0.10, left=0.22, right=0.98, hspace=0.04)

    fig.savefig(output, dpi=220)
    plt.close(fig)
    return 0


def plot_flooding(args: argparse.Namespace) -> int:
    payload = load_json(args.input)
    reports = payload["set_reports"]
    if not reports:
        raise SystemExit("The flooding JSON did not contain any set reports.")

    unit_suffix = f" ({args.condition_unit})" if args.condition_unit else ""
    condition_values = np.array([float(report["barrier"]) for report in reports], dtype=float)
    kobs = np.array([float(report["k_obs"]) for report in reports], dtype=float)
    log_kobs = np.array([float(report["log_k_obs"]) for report in reports], dtype=float)
    ln_acceleration = np.array([float(report["ln_exp_beta_v"]) for report in reports], dtype=float)
    gamma = float(payload["gamma"])
    logk0 = float(payload["logk0"])

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    plt = pyplot()

    fig, ax = plt.subplots(figsize=(3.35, 2.23), constrained_layout=True)
    ax.plot(condition_values, log_kobs, marker="o", color=BLUE)
    for report, xval, yval in zip(reports, condition_values, log_kobs):
        ax.annotate(str(report["barrier"]), (xval, yval), textcoords="offset points", xytext=(4, 4), fontsize=8, color=GRAY)
    ax.set_xlabel(f"{args.condition_label}{unit_suffix}")
    ax.set_ylabel(r"Observed ln($k_{\mathrm{obs}}$ / s$^{-1}$)")
    style_axis(ax)
    fig.savefig(f"{prefix}_observed_rate.png", dpi=220)
    plt.close(fig)

    xfit = np.linspace(float(np.min(ln_acceleration)) * 0.98, float(np.max(ln_acceleration)) * 1.02, 200)
    yfit = logk0 + gamma * xfit
    fig, ax = plt.subplots(figsize=(3.35, 2.23), constrained_layout=True)
    ax.plot(ln_acceleration, log_kobs, marker="o", linestyle="none", label="Simulation sets", color=BLUE)
    ax.plot(xfit, yfit, color=BLACK, label=fr"fit: ln($k_{{obs}}$) = ln($k_0$) + $\gamma$ ln($\alpha$)")
    for report, xval, yval in zip(reports, ln_acceleration, log_kobs):
        ax.annotate(str(report["barrier"]), (xval, yval), textcoords="offset points", xytext=(4, 4), fontsize=8, color=GRAY)
    ax.text(
        0.03,
        0.97,
        f"slope (gamma) = {gamma:.3f}\nintercept ln(k0 / s^-1) = {logk0:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": GRAY, "linewidth": 0.6},
    )
    ax.set_xlabel(r"ln acceleration factor, ln($\alpha$)")
    ax.set_ylabel(r"ln($k_{\mathrm{obs}}$ / s$^{-1}$)")
    style_axis(ax)
    ax.legend(loc="lower right", handlelength=1.5)
    fig.savefig(f"{prefix}_ln_kobs_vs_acceleration.png", dpi=220)
    plt.close(fig)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "regular-series":
        return plot_regular_series(args)
    if args.mode == "flooding":
        return plot_flooding(args)
    raise SystemExit(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
