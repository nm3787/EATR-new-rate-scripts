from __future__ import annotations

import argparse
import random
import sys

import numpy as np
from scipy import optimize
from scipy.stats import ks_1samp

import ks_censored as ksc
import rate_methods_library as RM


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    barr = parser.add_mutually_exclusive_group()
    temperature = parser.add_mutually_exclusive_group()
    event_find = parser.add_mutually_exclusive_group()
    parser.add_argument("-i", "--input", type=str, action="append", help="the files for a simulation set (call once for each set, i.e. -i path/to/1/*.colvar -i path/to/2/*.colvar etc.)", nargs="+")
    barr.add_argument("--barrier", type=np.float64, action="append", help="the BARRIER parameter in PLUMED for the simulation set (i.e. -i path/to/1/*.colvar --barrier 1 -i path/to/2/*.colvar --barrier 2 etc.)")
    barr.add_argument("--barriers", type=np.float64, help="the BARRIER parameter in PLUMED for each simulation set, defined all at once (i.e. -i path/to/1/*.colvar -i path/to/2/*.colvar etc. --barriers 1 2 etc.)", nargs="+")
    temperature.add_argument("--temp", type=np.float64, default=298, help="the temperature (in Kelvin) that the simulation was run at (make sure that ENERGYUNIT is correct)")
    temperature.add_argument("--kt", type=np.float64, default=None, help="the temperature (in kBT) that the simulation was run at")
    temperature.add_argument("--beta", type=np.float64, default=None, help="the inverse temperature 1/kBT that the simulation was run at")
    parser.add_argument("--tcol", type=int, default=0, help="the time column index in the COLVAR file")
    parser.add_argument("--vcol", type=int, default=2, help="the bias column index in the COLVAR file")
    parser.add_argument("--acol", type=int, default=None, help="the acceleration factor column index in the COLVAR file (only useful if OPESF is set)")
    parser.add_argument("--timeunit", type=np.float64, default=1e-12, help="the conversion factor from the time unit used in PLUMED to seconds")
    parser.add_argument("--energyunit", type=np.float64, default=1, help="the conversion factor from the energy unit used in PLUMED to kJ/mol (only needed if temperature was given in Kelvin)")
    parser.add_argument("--gammamin", type=np.float64, default=0, help="the minimum value of gamma to be checked")
    parser.add_argument("--gammamax", type=np.float64, default=1, help="the maximum value of gamma to be checked")
    parser.add_argument("--avgover", type=np.float64, default=-1, help="only include Vi(t) from the first transition time to when this fraction of transitioned simulations have transitioned in the set (-1 to include all data, 1 to only exclude Vi(t) before first transition)")
    parser.add_argument("--seed", type=int, default=None, help="the random number generator seed to use (for repeatability)")
    event_find.add_argument("--maxlen", type=int, default=None, help="the maximum number of rows in each COLVAR file before the simulation runs out of time")
    event_find.add_argument("--maxtime", type=np.float64, default=None, help="the maximum time that can appear in each COLVAR file (try to make it slightly less for floating point reasons)")
    event_find.add_argument("--numevents", type=int, default=None, action="append", help="the number of simulations that transitioned for each simulation set (i.e. -i path/to/1/*.colvar --numevents 20 -i path/to/2/*.colvar --numevents 18 etc.)")
    event_find.add_argument("--logfiles", type=str, default=None, action="append", help="the name of the file that contains the PLUMED log for each simulation in each set (i.e. -i path/to/1/*.colvar --logfiles path/to/1/*.log -i path/to/2/*.colvar --logfiles path/to/2/*.log etc.). Use check_order.py to make sure that the correct COLVAR files are paired with the correct log files.", nargs="+")
    parser.add_argument("-b", "--bootstrap", action="store_true", help="calculate errorbars with bootstrap analysis")
    parser.add_argument("--numboots", type=int, default=100, help="the number of bootstrap samples to use in bootsrapping if enabled")
    parser.add_argument("-q", "--quiet", action="store_true", help="do not print the results to the terminal as they are calculated")
    parser.add_argument("--cdf", action="store_true", help="estimate the biased observed rates using CDF fitting (not recommended if you have arbitrarily right-censored data, such as simulations being killed before reaching max steps)")
    parser.add_argument("--timefirst", action="store_true", help="estimate ln<e^βγV> by averaging over time for each simulation, then over the simulations (default is over simulations first)")
    parser.add_argument("--nooffset", action="store_true", help="do not add the BARRIER parameter to the bias (OPES simulations in PLUMED offset the bias by -1*BARRIER, so do not use this for such simulations)")
    parser.add_argument("--opesf", action="store_true", help="also run the OPES flooding analysis on all of the data")
    return parser


def parse_beta(args: argparse.Namespace) -> float:
    if args.beta is not None:
        beta = args.beta
        if not args.quiet:
            print(f"Using β = {beta}")
        return beta
    if args.kt is not None:
        beta = 1 / args.kt
        if not args.quiet:
            print(f"Using β = 1/kBT = {beta}")
        return beta
    beta = args.energyunit / (8.314462618e-3 * args.temp)
    if not args.quiet:
        print(f"Using β = 1/kBT = {beta}, with PLUMED energy unit equivalent to {args.energyunit} kJ/mol")
    return beta


def analyze(args: argparse.Namespace) -> tuple[float, float, float | None]:
    beta = parse_beta(args)
    random.seed(args.seed)

    barriers = args.barriers if args.barriers is not None else args.barrier
    if args.input is None or len(args.input) < 2:
        raise SystemExit("You need at least two sets of simulations, each with a different value for BARRIER in PLUMED.")
    if barriers is None or len(args.input) != len(barriers):
        raise SystemExit(f"You must specify the same number of BARRIER values as simulation sets in INPUT. There are {len(barriers) if barriers is not None else 0} BARRIER values and {len(args.input)} simulation sets.")

    num_eventss = args.numevents if args.numevents is not None else [None] * len(barriers)
    log_filess = args.logfiles if args.logfiles is not None else [None] * len(barriers)
    gamma_bounds = (args.gammamin, args.gammamax)
    axis_first = 1 if args.timefirst else 0

    datas = [RM.get_data(colvars, args.tcol, args.vcol, acc_col=args.acol, time_scale_factor=args.timeunit) for colvars in args.input]
    events = [RM.get_event(datas[i], maxlen=args.maxlen, maxtime=args.maxtime, num_events=num_eventss[i], log_files=log_filess[i], quiet=True) for i in range(len(datas))]

    def analyze_indices(indicess: list[list[int]], quiet: bool = False) -> tuple[float, float, float | None]:
        logk0_opesf = None
        opesf_times: list[float] = []
        opesf_event: list[bool] = []
        v_datas: dict[float, np.ndarray] = {}
        obs_rates: dict[float, float] = {}

        for i, barrier in enumerate(barriers):
            barrier_add = 0 if args.nooffset else barrier
            data = [datas[i][j] for j in indicess[i]]
            event = np.array([events[i][j] for j in indicess[i]])
            if not quiet:
                print(f"Simulation Set: BARRIER = {beta * barrier} kBT")
                print(f"{event.sum()} out of {len(data)} simulations transitioned.")
                max_biases = [np.max(traj[:, 1] + barrier) for traj in data]
                print(f"avg. max. bias: {np.mean(max_biases)}")

            colvar_row_counts = np.sort([len(traj[:, 0]) for traj in data])
            max_index = colvar_row_counts[-1] if args.avgover < 0 else colvar_row_counts[int(abs(args.avgover) * np.sum(event))]
            min_index = 0 if args.avgover < 0 else colvar_row_counts[0]
            v_data = np.full((len(data), max_index - min_index), np.nan)
            for traj_index, traj in enumerate(data):
                v_data[traj_index, : (min(len(traj), max_index) - min_index)] = traj[min_index:max_index, 1] + barrier_add
            v_datas[barrier] = v_data

            final_times = np.array([traj[-1, 0] for traj in data])
            if args.opesf:
                rescaled_times = RM.iMetaD_rescaled_times(data, beta, bias_shift=barrier_add)
                opesf_times.extend(list(rescaled_times))
                opesf_event.extend(list(event))

            ecdfxs = np.sort(final_times)
            ecdfys = np.linspace(1 / len(event), 1, len(event))
            emp_rate = event.sum() / final_times.sum()
            if args.cdf:
                obs_rate = optimize.curve_fit(lambda t, k: 1 - np.exp(-k * t), ecdfxs[: event.sum()], ecdfys[: event.sum()], p0=emp_rate)[0][0]
                obs_rates[barrier] = obs_rate
                ks_stat, p = ks_1samp(ecdfxs[: event.sum()], lambda t: 1 - np.exp(-obs_rate * t))
            else:
                obs_rate = emp_rate
                obs_rates[barrier] = emp_rate
                ks_stat, p = ksc.ks_1samp_censored(final_times, event, lambda t: np.exp(-emp_rate * t))

            if not quiet:
                print(f"tau_obs: {1 / obs_rate}, k_obs: {obs_rate}, log k_obs: {np.log(obs_rate)}")
                print(f"KS stat: {ks_stat}; p = {p}")
                avg = np.mean(np.nanmean(np.exp(beta * v_data), axis=0))
                print(rf"ln<e^βV>: {np.log(avg)}")
                print("")

        if args.opesf:
            logk0_opesf = np.log(RM.iMetaD_FitCDF_times(np.array(opesf_times), event=np.array(opesf_event)))

        def variance(gamma: float) -> float:
            logk0s = []
            for barrier in barriers:
                avg = np.mean(np.nanmean(np.exp(beta * gamma * v_datas[barrier]), axis=axis_first))
                logk0s.append(np.log(obs_rates[barrier]) - np.log(avg))
            return np.var(logk0s)

        gamma_best = optimize.minimize_scalar(variance, bounds=gamma_bounds).x
        logk0s = []
        for barrier in barriers:
            avg = np.mean(np.nanmean(np.exp(beta * gamma_best * v_datas[barrier]), axis=axis_first))
            logk0s.append(np.log(obs_rates[barrier]) - np.log(avg))
        logk0_best = np.mean(logk0s)
        return logk0_best, gamma_best, logk0_opesf

    if not args.bootstrap:
        result = analyze_indices([list(range(len(data))) for data in datas], quiet=args.quiet)
        logk0_best, gamma_best, logk0_opes = result
        print(f"\nk0: {np.exp(logk0_best)} s^-1, logk0: {logk0_best} s^-1, τ0: {np.exp(-logk0_best)} s, gamma: {gamma_best}{', OPES logk0: ' + str(logk0_opes) + ' s^-1' if args.opesf else ''}")
        return result

    sample_logk0 = []
    sample_gamma = []
    sample_opesf = []
    for i in range(args.numboots):
        indicess = [random.choices(list(range(len(data))), k=len(data)) for data in datas]
        logk0, gamma, logk0_opesf = analyze_indices(indicess, quiet=True)
        sample_logk0.append(logk0)
        sample_gamma.append(gamma)
        sample_opesf.append(logk0_opesf)
        if not args.quiet:
            print(i)
    print(f"logk0: {np.mean(sample_logk0)} +/- σ {np.std(sample_logk0)} s^-1, τ0: {np.exp(-np.mean(sample_logk0))} s, gamma: {np.mean(sample_gamma)} +/- σ {np.std(sample_gamma)}{', OPES logk0: ' + str(np.mean(sample_opesf)) + ' +/- σ ' + str(np.std(sample_opesf)) + ' s^-1' if sample_opesf[0] is not None else ''}")
    return float(np.mean(sample_logk0)), float(np.mean(sample_gamma)), None if sample_opesf[0] is None else float(np.mean(sample_opesf))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
