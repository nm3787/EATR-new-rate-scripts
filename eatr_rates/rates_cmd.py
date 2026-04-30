from __future__ import annotations

import argparse
import json
import random

import numpy as np
from scipy.stats import gamma as gamma_func
from scipy.stats import ks_1samp, ks_2samp

import rate_methods_library as RM

bopt_avail = False
try:
    from bayes_opt import BayesianOptimization as bopt
    from bayes_opt import acquisition

    bopt_avail = True
except Exception:
    bopt_avail = False

boots_avail = False
try:
    from scipy.stats import bootstrap as bootstr

    boots_avail = True
except Exception:
    boots_avail = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    temperature = parser.add_mutually_exclusive_group()
    event_find = parser.add_mutually_exclusive_group()
    parser.add_argument("-i", "--input", type=str, help="the simulation COLVAR files to analyze", nargs="+")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="rates.json",
        help="the name of the output JSON file (DEFAULT: rates.json)",
    )
    temperature.add_argument(
        "--temp",
        type=np.float64,
        default=298,
        help="the temperature (in Kelvin) that the simulation was run at (make sure that ENERGYUNIT is correct) (DEFAULT: 298K)",
    )
    temperature.add_argument("--kt", type=np.float64, default=None, help="the temperature (in kBT) that the simulation was run at")
    temperature.add_argument("--beta", type=np.float64, default=None, help="the inverse temperature 1/kBT that the simulation was run at")
    parser.add_argument("--tcol", type=int, default=0, help="the time column index in the COLVAR file. (DEFAULT: 0)")
    parser.add_argument("--vcol", type=int, default=2, help="the bias column index in the COLVAR file. (DEFAULT: 2)")
    parser.add_argument("--acol", type=int, default=None, help="the acceleration factor column index in the COLVAR file, if present. (DEFAULT: None)")
    parser.add_argument("--mcol", type=int, default=None, help="the max bias column index in the COLVAR file, if present. (DEFAULT: None)")
    parser.add_argument(
        "--timeunit",
        type=np.float64,
        default=1e-12,
        help="the conversion factor from the time unit used in PLUMED to seconds (DEFAULT: 1e-12, for picoseconds)",
    )
    parser.add_argument(
        "--energyunit",
        type=np.float64,
        default=1,
        help="the conversion factor from the energy unit used in PLUMED to kJ/mol (only needed if temperature was given in Kelvin) (DEFAULT: 1, for kJ/mol)",
    )
    parser.add_argument(
        "--barrier",
        type=np.float64,
        default=0,
        help="the BARRIER parameter in PLUMED for OPES (it is not a good idea to use this script to run KTR and EATR on OPES simulations btw) (DEFAULT: 0)",
    )
    parser.add_argument("--gammamin", type=np.float64, default=0, help="the minimum value of gamma to be checked in KTR and EATR (DEFAULT: 0)")
    parser.add_argument("--gammamax", type=np.float64, default=1, help="the maximum value of gamma to be checked in KTR and EATR (DEFAULT: 1)")
    parser.add_argument("--kmin", type=np.float64, default=-np.inf, help="the minimum value of k0 to be checked in CDF fitting (DEFAULT: -inf)")
    parser.add_argument("--kmax", type=np.float64, default=np.inf, help="the maximum value of k0 to be checked in CDF fitting (DEFAULT: inf)")
    parser.add_argument("--seed", type=int, default=None, help="the random number generator seed to use (for repeatability) (DEFAULT: None)")
    parser.add_argument("--cores", type=int, default=1, help="the number of cores for multiprocessing (DEFAULT: 1, no multiprocessing)")
    event_find.add_argument(
        "--maxlen",
        type=int,
        default=None,
        help="the maximum number of rows in each COLVAR file before the simulation runs out of time (DEFAULT: Do not use this to determine which simulations transitioned.)",
    )
    event_find.add_argument(
        "--maxtime",
        type=np.float64,
        default=None,
        help="the maximum time that can appear in each COLVAR file (try to make it slightly less for floating point reasons) (DEFAULT: Do not use this to determine which simulations transitioned.)",
    )
    event_find.add_argument(
        "--numevents",
        type=int,
        default=None,
        help="the number of simulations that transitioned (DEFAULT: Do not use this to determine which simulations transitioned.)",
    )
    event_find.add_argument(
        "--logfiles",
        type=str,
        default=None,
        help="the files that contains the PLUMED logs. Use check_order.py to make sure that the correct COLVAR files are paired with the correct log files (DEFAULT: Do not use this to determine which simulations transitioned.)",
        nargs="+",
    )
    parser.add_argument("-m", "--imetadmle", action="store_true", help="run the Tiwary rate estimator for infrequent metadynamics")
    parser.add_argument("-M", "--imetadcdf", action="store_true", help="run the Salvalaglio rate estimator for infrequent metadynamics")
    parser.add_argument("-k", "--ktrmle", action="store_true", help="run original KTR method")
    parser.add_argument("-K", "--ktrcdf", action="store_true", help="run KTR method estimating gamma and k0 with CDF")
    parser.add_argument("-e", "--eatrmle", action="store_true", help="run EATR method estimating gamma and k0 with likelihood")
    parser.add_argument("-E", "--eatrcdf", action="store_true", help="run EATR method estimating gamma and k0 with CDF")
    parser.add_argument("-b", "--bootstrap", action="store_true", help="calculate errorbars with bootstrap analysis")
    parser.add_argument("--std", action="store_true", help="use standard deviations in bootstrap analysis even if SciPy has the bootstrap method")
    parser.add_argument("--numboots", type=int, default=100, help="the number of bootstrap samples to use in bootsrapping if enabled (DEFAULT: 100)")
    parser.add_argument("-B", "--bayesopt", action="store_true", help="use Bayseian Optimization algorithm for optimizing if available")
    parser.add_argument("-l", "--logtrick", action="store_true", help="use log-sum-exp trick to potentially increase precision (generally unneeded)")
    parser.add_argument("-q", "--quiet", action="store_true", help="do not print the results to the terminal as they are calculated")
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
    beta = args.energyunit / (0.008314 * args.temp)
    if not args.quiet:
        print(f"Using β = 1/kBT = {beta}, with PLUMED energy unit equivalent to {args.energyunit} kJ/mol")
    return beta


def validate_args(args: argparse.Namespace) -> None:
    if not (args.imetadmle or args.imetadcdf or args.ktrmle or args.ktrcdf or args.eatrmle or args.eatrcdf):
        raise SystemExit("Specify at least one rate method to perform from -m -M -k -K -e -E (M=iMetaD, K=KTR, E=EATR; lowercase is MLE and uppercase is CDF).")


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "low") and hasattr(value, "high"):
        return {"low": json_ready(value.low), "high": json_ready(value.high)}
    return value


def analyze(args: argparse.Namespace) -> dict[str, object]:
    global boots_avail

    beta = parse_beta(args)
    validate_args(args)

    gamma_bounds = (args.gammamin, args.gammamax)
    k_bounds = (args.kmin, args.kmax)

    if args.bayesopt and bopt_avail:
        if not args.quiet:
            print("Bayesian Optimization module activated.")
    elif args.bayesopt:
        print("The Bayesian Optimization module was not able to be loaded. Defaulting to local optimizers.")

    if args.std:
        boots_avail = False
    if not args.quiet:
        if args.bootstrap and boots_avail:
            print("Bootstrapping is activated. Will use SciPy bootstrap method (errors are 95% confidence intervals).")
        elif args.bootstrap:
            print("SciPy bootstrap method is not available. Will use internal bootstrap method (errors are standard deviations).")

    if args.barrier > 0 and (args.ktrmle or args.ktrcdf or args.eatrmle or args.eatrcdf):
        print("WARNING: Running KTR and/or EATR on OPES simulations using this analysis script is not expected to work properly! You should instead use the EATR-OPES method (not published yet).")

    seed = args.seed
    random.seed(seed)
    seed = seed if seed is None else seed + 1

    results: dict[str, object] = {}
    data = RM.get_data(args.input, args.tcol, args.vcol, acc_col=args.acol, maxbias_col=args.mcol, time_scale_factor=args.timeunit)
    event = RM.get_event(data, maxlen=args.maxlen, maxtime=args.maxtime, num_events=args.numevents, log_files=args.logfiles, quiet=args.quiet)

    if args.imetadmle or args.imetadcdf:
        rescaled_times = RM.iMetaD_rescaled_times(data, beta, bias_shift=args.barrier)

    if args.imetadmle:
        if not args.bootstrap:
            results["iMetaD MLE ln k"] = np.log(RM.iMetaD_invMRT_times(rescaled_times, event=event))
        else:
            if boots_avail:
                indices = list(range(len(data)))
                results["iMetaD MLE ln k"] = np.log(RM.iMetaD_invMRT_times(rescaled_times, event=event))
                res = bootstr(
                    (indices,),
                    lambda idxs: np.log(RM.iMetaD_invMRT([data[idx] for idx in idxs], beta, event=np.array([event[idx] for idx in idxs]), bias_shift=args.barrier)),
                    random_state=seed,
                    vectorized=False,
                    n_resamples=args.numboots,
                )
                seed = seed if seed is None else seed + 1
                results["iMetaD MLE ln k CI"] = res.confidence_interval
            else:
                sample = RM.bootstrap(data, lambda subset, eve: RM.iMetaD_invMRT(subset, beta, event=eve, bias_shift=args.barrier), args.numboots, event=event, return_stat=True)
                results["iMetaD MLE ln k"] = np.mean(np.log(sample))
                results["iMetaD MLE ln k std"] = np.std(np.log(sample))
        size = np.int64(len(data) * 5e4)
        rvs1 = gamma_func.rvs(1, scale=np.exp(-results["iMetaD MLE ln k"]), size=size, random_state=seed)
        seed = seed if seed is None else seed + 1
        ks_stat, p = ks_2samp(rvs1, rescaled_times[event])
        results["iMetaD MLE KS stat"] = ks_stat
        results["iMetaD MLE p value"] = p
        if not args.quiet:
            if not args.bootstrap:
                print(f'iMetaD MLE: lnk0 = {results["iMetaD MLE ln k"]} (s^-1); KS: {ks_stat}, p = {p}')
            elif boots_avail:
                print(f'iMetaD MLE: lnk0 = {results["iMetaD MLE ln k"]} (s^-1), 95% CI: {results["iMetaD MLE ln k CI"][0]} to {results["iMetaD MLE ln k CI"][1]}; KS: {ks_stat}, p = {p}')
            else:
                print(f'iMetaD MLE: lnk0 = {results["iMetaD MLE ln k"]} +/- {results["iMetaD MLE ln k std"]} (s^-1); KS: {ks_stat}, p = {p}')

    if args.imetadcdf:
        if not args.bootstrap:
            results["iMetaD CDF ln k"] = np.log(RM.iMetaD_FitCDF_times(rescaled_times, event=event, k_bounds=k_bounds))
        else:
            if boots_avail:
                indices = list(range(len(data)))
                results["iMetaD CDF ln k"] = np.log(RM.iMetaD_FitCDF_times(rescaled_times, event=event, k_bounds=k_bounds))
                res = bootstr(
                    (indices,),
                    lambda idxs: np.log(RM.iMetaD_FitCDF([data[idx] for idx in idxs], beta, event=np.array([event[idx] for idx in idxs]), bias_shift=args.barrier, k_bounds=k_bounds)),
                    random_state=seed,
                    vectorized=False,
                    n_resamples=args.numboots,
                )
                seed = seed if seed is None else seed + 1
                results["iMetaD CDF ln k CI"] = res.confidence_interval
            else:
                sample = RM.bootstrap(data, lambda subset, eve: RM.iMetaD_FitCDF(subset, beta, event=eve, bias_shift=args.barrier), args.numboots, event=event, return_stat=True)
                results["iMetaD CDF ln k"] = np.mean(np.log(sample))
                results["iMetaD CDF ln k std"] = np.std(np.log(sample))
        size = np.int64(len(data) * 5e4)
        rvs1 = gamma_func.rvs(1, scale=np.exp(-results["iMetaD CDF ln k"]), size=size, random_state=seed)
        seed = seed if seed is None else seed + 1
        ks_stat, p = ks_2samp(rvs1, rescaled_times[event])
        results["iMetaD CDF KS stat"] = ks_stat
        results["iMetaD CDF p value"] = p
        if not args.quiet:
            if not args.bootstrap:
                print(f'iMetaD CDF: lnk0 = {results["iMetaD CDF ln k"]} (s^-1); KS: {ks_stat}, p = {p}')
            elif boots_avail:
                print(f'iMetaD CDF: lnk0 = {results["iMetaD CDF ln k"]} (s^-1), 95% CI: {results["iMetaD CDF ln k CI"][0]} to {results["iMetaD CDF ln k CI"][1]}; KS: {ks_stat}, p = {p}')
            else:
                print(f'iMetaD CDF: lnk0 = {results["iMetaD CDF ln k"]} +/- {results["iMetaD CDF ln k std"]} (s^-1); KS: {ks_stat}, p = {p}')

    final_time_indices = np.array([int(len(traj) - 1) for traj in data])
    if args.ktrmle or args.ktrcdf:
        vmb_average = RM.avg_max_bias(data, beta, bias_shift=args.barrier)

    if args.ktrmle:
        if not args.bootstrap:
            result = RM.KTR_MLE_rate_VMB(vmb_average, final_time_indices, event=event, gamma_bounds=gamma_bounds, cores=args.cores, logTrick=args.logtrick, do_bopt=args.bayesopt)
            results["KTR MLE ln k"] = np.log(result[0])
            results["KTR MLE gamma"] = result[1]
        else:
            if boots_avail:
                indices = list(range(len(data)))
                result = RM.KTR_MLE_rate_VMB(vmb_average, final_time_indices, event=event, gamma_bounds=gamma_bounds, cores=args.cores, logTrick=args.logtrick, do_bopt=args.bayesopt)
                res = bootstr(
                    (indices,),
                    lambda idxs: RM.KTR_MLE_rate([data[idx] for idx in idxs], beta, event=np.array([event[idx] for idx in idxs]), gamma_bounds=gamma_bounds, cores=args.cores, logTrick=args.logtrick, do_bopt=args.bayesopt, bias_shift=args.barrier),
                    random_state=seed,
                    vectorized=False,
                    n_resamples=args.numboots,
                )
                seed = seed if seed is None else seed + 1
                results["KTR MLE ln k"] = np.log(result[0])
                results["KTR MLE gamma"] = result[1]
                results["KTR MLE ln k CI"] = [np.log(res.confidence_interval.low[0]), np.log(res.confidence_interval.high[0])]
                results["KTR MLE gamma CI"] = [res.confidence_interval.low[1], res.confidence_interval.high[1]]
            else:
                sample = RM.bootstrap(data, lambda subset, eve: RM.KTR_MLE_rate(subset, beta, event=eve, gamma_bounds=gamma_bounds, cores=args.cores, logTrick=args.logtrick, do_bopt=args.bayesopt, bias_shift=args.barrier), args.numboots, double=True, event=event, return_stat=True)
                results["KTR MLE ln k"] = np.mean(np.log(sample[:, 0]))
                results["KTR MLE gamma"] = np.mean(sample[:, 1])
                results["KTR MLE ln k std"] = np.std(np.log(sample[:, 0]))
                results["KTR MLE gamma std"] = np.std(sample[:, 1])
        ks_stat, p = ks_1samp(final_time_indices[event], lambda idx: RM.KTR_CDF(idx, np.exp(results["KTR MLE ln k"]), results["KTR MLE gamma"], vmb_average, cores=args.cores, logTrick=args.logtrick))
        results["KTR MLE KS stat"] = ks_stat
        results["KTR MLE p value"] = p
        if not args.quiet:
            if not args.bootstrap:
                print(f'KTR MLE: lnk0 = {results["KTR MLE ln k"]} (s^-1), γ = {results["KTR MLE gamma"]}; KS: {ks_stat}, p = {p}')
            elif boots_avail:
                print(f'KTR MLE: lnk0 = {results["KTR MLE ln k"]} (s^-1), 95% CI: {results["KTR MLE ln k CI"][0]} to {results["KTR MLE ln k CI"][1]}, γ = {results["KTR MLE gamma"]}, 95% CI: {results["KTR MLE gamma CI"][0]} to {results["KTR MLE gamma CI"][1]}; KS: {ks_stat}, p = {p}')
            else:
                print(f'KTR MLE: lnk0 = {results["KTR MLE ln k"]} +/- {results["KTR MLE ln k std"]} (s^-1), γ = {results["KTR MLE gamma"]} +/- {results["KTR MLE gamma std"]}; KS: {ks_stat}, p = {p}')

    if args.ktrcdf:
        if not args.bootstrap:
            result = RM.KTR_CDF_rate_VMB(vmb_average, final_time_indices, event=event, k_bounds=k_bounds, gamma_bounds=gamma_bounds, cores=args.cores, logTrick=args.logtrick, do_bopt=args.bayesopt)
            results["KTR CDF ln k"] = np.log(result[0])
            results["KTR CDF gamma"] = result[1]
        else:
            if boots_avail:
                indices = list(range(len(data)))
                result = RM.KTR_CDF_rate_VMB(vmb_average, final_time_indices, event=event, k_bounds=k_bounds, gamma_bounds=gamma_bounds, cores=args.cores, logTrick=args.logtrick, do_bopt=args.bayesopt)
                res = bootstr(
                    (indices,),
                    lambda idxs: RM.KTR_CDF_rate([data[idx] for idx in idxs], beta, event=np.array([event[idx] for idx in idxs]), k_bounds=k_bounds, gamma_bounds=gamma_bounds, cores=args.cores, logTrick=args.logtrick, do_bopt=args.bayesopt, bias_shift=args.barrier),
                    random_state=seed,
                    vectorized=False,
                    n_resamples=args.numboots,
                )
                seed = seed if seed is None else seed + 1
                results["KTR CDF ln k"] = np.log(result[0])
                results["KTR CDF gamma"] = result[1]
                results["KTR CDF ln k CI"] = [np.log(res.confidence_interval.low[0]), np.log(res.confidence_interval.high[0])]
                results["KTR CDF gamma CI"] = [res.confidence_interval.low[1], res.confidence_interval.high[1]]
            else:
                sample = RM.bootstrap(data, lambda subset, eve: RM.KTR_CDF_rate(subset, beta, event=eve, k_bounds=k_bounds, gamma_bounds=gamma_bounds, cores=args.cores, logTrick=args.logtrick, do_bopt=args.bayesopt, bias_shift=args.barrier), args.numboots, double=True, event=event, return_stat=True)
                results["KTR CDF ln k"] = np.mean(np.log(sample[:, 0]))
                results["KTR CDF gamma"] = np.mean(sample[:, 1])
                results["KTR CDF ln k std"] = np.std(np.log(sample[:, 0]))
                results["KTR CDF gamma std"] = np.std(sample[:, 1])
        ks_stat, p = ks_1samp(final_time_indices[event], lambda idx: RM.KTR_CDF(idx, np.exp(results["KTR CDF ln k"]), results["KTR CDF gamma"], vmb_average, cores=args.cores, logTrick=args.logtrick))
        results["KTR CDF KS stat"] = ks_stat
        results["KTR CDF p value"] = p
        if not args.quiet:
            if not args.bootstrap:
                print(f'KTR CDF: lnk0 = {results["KTR CDF ln k"]} (s^-1), γ = {results["KTR CDF gamma"]}; KS: {ks_stat}, p = {p}')
            elif boots_avail:
                print(f'KTR CDF: lnk0 = {results["KTR CDF ln k"]} (s^-1), 95% CI: {results["KTR CDF ln k CI"][0]} to {results["KTR CDF ln k CI"][1]}, γ = {results["KTR CDF gamma"]}, 95% CI: {results["KTR CDF gamma CI"][0]} to {results["KTR CDF gamma CI"][1]}; KS: {ks_stat}, p = {p}')
            else:
                print(f'KTR CDF: lnk0 = {results["KTR CDF ln k"]} +/- {results["KTR CDF ln k std"]} (s^-1), γ = {results["KTR CDF gamma"]} +/- {results["KTR CDF gamma std"]}; KS: {ks_stat}, p = {p}')

    if args.eatrmle:
        if not args.bootstrap:
            result = RM.EATR_MLE_rate(data, beta, event=event, gamma_bounds=gamma_bounds, cores=args.cores, logTrick=args.logtrick, do_bopt=args.bayesopt, bias_shift=args.barrier)
            results["EATR MLE ln k"] = np.log(result[0])
            results["EATR MLE gamma"] = result[1]
        else:
            if boots_avail:
                indices = list(range(len(data)))
                result = RM.EATR_MLE_rate(data, beta, event=event, gamma_bounds=gamma_bounds, cores=args.cores, logTrick=args.logtrick, do_bopt=args.bayesopt, bias_shift=args.barrier)
                res = bootstr(
                    (indices,),
                    lambda idxs: RM.EATR_MLE_rate([data[idx] for idx in idxs], beta, event=np.array([event[idx] for idx in idxs]), gamma_bounds=gamma_bounds, cores=args.cores, logTrick=args.logtrick, do_bopt=args.bayesopt, bias_shift=args.barrier),
                    random_state=seed,
                    vectorized=False,
                    n_resamples=args.numboots,
                )
                seed = seed if seed is None else seed + 1
                results["EATR MLE ln k"] = np.log(result[0])
                results["EATR MLE gamma"] = result[1]
                results["EATR MLE ln k CI"] = [np.log(res.confidence_interval.low[0]), np.log(res.confidence_interval.high[0])]
                results["EATR MLE gamma CI"] = [res.confidence_interval.low[1], res.confidence_interval.high[1]]
            else:
                sample = RM.bootstrap(data, lambda subset, eve: RM.EATR_MLE_rate(subset, beta, event=eve, gamma_bounds=gamma_bounds, cores=args.cores, logTrick=args.logtrick, do_bopt=args.bayesopt, bias_shift=args.barrier), args.numboots, double=True, event=event, return_stat=True)
                results["EATR MLE ln k"] = np.mean(np.log(sample[:, 0]))
                results["EATR MLE gamma"] = np.mean(sample[:, 1])
                results["EATR MLE ln k std"] = np.std(np.log(sample[:, 0]))
                results["EATR MLE gamma std"] = np.std(sample[:, 1])
        log_average_exp = RM.avg_exponential(data, beta, results["EATR MLE gamma"], bias_shift=args.barrier)
        ks_stat, p = ks_1samp(final_time_indices[event], lambda idx: RM.EATR_CDF(idx, np.exp(results["EATR MLE ln k"]), log_average_exp, cores=args.cores, logTrick=args.logtrick))
        results["EATR MLE KS stat"] = ks_stat
        results["EATR MLE p value"] = p
        if not args.quiet:
            if not args.bootstrap:
                print(f'EATR MLE: lnk0 = {results["EATR MLE ln k"]} (s^-1), γ = {results["EATR MLE gamma"]}; KS: {ks_stat}, p = {p}')
            elif boots_avail:
                print(f'EATR MLE: lnk0 = {results["EATR MLE ln k"]} (s^-1), 95% CI: {results["EATR MLE ln k CI"][0]} to {results["EATR MLE ln k CI"][1]}, γ = {results["EATR MLE gamma"]}, 95% CI: {results["EATR MLE gamma CI"][0]} to {results["EATR MLE gamma CI"][1]}; KS: {ks_stat}, p = {p}')
            else:
                print(f'EATR MLE: lnk0 = {results["EATR MLE ln k"]} +/- {results["EATR MLE ln k std"]} (s^-1), γ = {results["EATR MLE gamma"]} +/- {results["EATR MLE gamma std"]}; KS: {ks_stat}, p = {p}')

    if args.eatrcdf:
        if not args.bootstrap:
            result = RM.EATR_CDF_rate(data, beta, event=event, k_bounds=k_bounds, gamma_bounds=gamma_bounds, cores=args.cores, logTrick=args.logtrick, do_bopt=args.bayesopt, bias_shift=args.barrier)
            results["EATR CDF ln k"] = np.log(result[0])
            results["EATR CDF gamma"] = result[1]
        else:
            if boots_avail:
                indices = list(range(len(data)))
                result = RM.EATR_CDF_rate(data, beta, event=event, k_bounds=k_bounds, gamma_bounds=gamma_bounds, cores=args.cores, logTrick=args.logtrick, do_bopt=args.bayesopt, bias_shift=args.barrier)
                res = bootstr(
                    (indices,),
                    lambda idxs: RM.EATR_CDF_rate([data[idx] for idx in idxs], beta, event=np.array([event[idx] for idx in idxs]), k_bounds=k_bounds, gamma_bounds=gamma_bounds, cores=args.cores, logTrick=args.logtrick, do_bopt=args.bayesopt, bias_shift=args.barrier),
                    random_state=seed,
                    vectorized=False,
                    n_resamples=args.numboots,
                )
                seed = seed if seed is None else seed + 1
                results["EATR CDF ln k"] = np.log(result[0])
                results["EATR CDF gamma"] = result[1]
                results["EATR CDF ln k CI"] = [np.log(res.confidence_interval.low[0]), np.log(res.confidence_interval.high[0])]
                results["EATR CDF gamma CI"] = [res.confidence_interval.low[1], res.confidence_interval.high[1]]
            else:
                sample = RM.bootstrap(data, lambda subset, eve: RM.EATR_CDF_rate(subset, beta, event=eve, k_bounds=k_bounds, gamma_bounds=gamma_bounds, cores=args.cores, logTrick=args.logtrick, do_bopt=args.bayesopt, bias_shift=args.barrier), args.numboots, double=True, event=event, return_stat=True)
                results["EATR CDF ln k"] = np.mean(np.log(sample[:, 0]))
                results["EATR CDF gamma"] = np.mean(sample[:, 1])
                results["EATR CDF ln k std"] = np.std(np.log(sample[:, 0]))
                results["EATR CDF gamma std"] = np.std(sample[:, 1])
        log_average_exp = RM.avg_exponential(data, beta, results["EATR CDF gamma"], bias_shift=args.barrier)
        ks_stat, p = ks_1samp(final_time_indices[event], lambda idx: RM.EATR_CDF(idx, np.exp(results["EATR CDF ln k"]), log_average_exp, cores=args.cores, logTrick=args.logtrick))
        results["EATR CDF KS stat"] = ks_stat
        results["EATR CDF p value"] = p
        if not args.quiet:
            if not args.bootstrap:
                print(f'EATR CDF: lnk0 = {results["EATR CDF ln k"]} (s^-1), γ = {results["EATR CDF gamma"]}; KS: {ks_stat}, p = {p}')
            elif boots_avail:
                print(f'EATR CDF: lnk0 = {results["EATR CDF ln k"]} (s^-1), 95% CI: {results["EATR CDF ln k CI"][0]} to {results["EATR CDF ln k CI"][1]}, γ = {results["EATR CDF gamma"]}, 95% CI: {results["EATR CDF gamma CI"][0]} to {results["EATR CDF gamma CI"][1]}; KS: {ks_stat}, p = {p}')
            else:
                print(f'EATR CDF: lnk0 = {results["EATR CDF ln k"]} +/- {results["EATR CDF ln k std"]} (s^-1), γ = {results["EATR CDF gamma"]} +/- {results["EATR CDF gamma std"]}; KS: {ks_stat}, p = {p}')

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(json_ready(results), handle)
    return results


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
