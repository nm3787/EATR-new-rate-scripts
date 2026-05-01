# EATR Rate Analysis

This repository provides command-line tools and Python modules for estimating unbiased rate constants from biased molecular dynamics simulations.

It supports:

- infrequent metadynamics / WT-MetaD style analyses
- OPES flooding analyses
- KTR and EATR estimators
- EATR-flooding across multiple sets of simulations with different bias strengths

The packaged commands are:

- `eatr-analysis`
- `eatr-flooding-analysis`
- `eatr-check-order`

## Theory

Rare-event kinetics are often estimated from biased simulations by relating the observed transition times under bias to an underlying unbiased rate constant `k0`.

This repository includes several related estimators:

- `iMetaD`
  Uses the standard infrequent metadynamics rescaling idea, where the observed time is accelerated by the bias.
- `KTR`
  Introduces a fitted efficiency parameter `γ` to account for the fact that the biased collective variable may not be an ideal reaction coordinate.
- `EATR`
  Uses an exponential average of the time-dependent bias to estimate both `k0` and `γ`. This is the main estimator introduced in Mazzaferro et al., JCTC 2024.
- `EATR-flooding`
  Extends the same idea to quasi-static or flooding-style biasing, especially OPES flooding, by comparing multiple sets of simulations performed with different bias strengths.

The relevant papers included in [papers](/Volumes/HockyExtraSpace/Dropbox/research/projects/NNP-EATR-data-analysis/EATR-new-rate-scripts/papers) are:

- [52_2024_Mazzaferro_EATR_JCTC.pdf](/Volumes/HockyExtraSpace/Dropbox/research/projects/NNP-EATR-data-analysis/EATR-new-rate-scripts/papers/52_2024_Mazzaferro_EATR_JCTC.pdf)
- [52_2024_Mazzaferro_EATR_JCTC_SI.pdf](/Volumes/HockyExtraSpace/Dropbox/research/projects/NNP-EATR-data-analysis/EATR-new-rate-scripts/papers/52_2024_Mazzaferro_EATR_JCTC_SI.pdf)
- [eatr-flooding-plusSI-arxiv.pdf](/Volumes/HockyExtraSpace/Dropbox/research/projects/NNP-EATR-data-analysis/EATR-new-rate-scripts/papers/eatr-flooding-plusSI-arxiv.pdf)

Practical guidance:

- Use `eatr-analysis` for time-dependent MetaD-style biasing.
- Use `eatr-flooding-analysis` for OPES flooding or any workflow where you intentionally vary the amount of bias across multiple simulation sets.
- You can also apply `eatr-flooding-analysis` to MetaD if you have multiple sets with different hill-deposition paces or other systematically varied biasing conditions.

## Installation

Install as a package:

```bash
pip install .
```

For development:

```bash
pip install -e ".[dev]"
```

If you want to reproduce the example plots in this repository, you will also need `matplotlib`.

## Command Overview

### `eatr-analysis`

This command analyzes one collection of trajectories from a single biasing protocol. It can compute:

- `iMetaD MLE`
- `iMetaD CDF`
- `KTR MLE`
- `KTR CDF`
- `EATR MLE`
- `EATR CDF`

Typical usage:

```bash
eatr-analysis -i run_*/*.colvar --temp 310 -E
```

Important arguments:

- `-i`, `--input`
  Input COLVAR files.
- `-o`, `--output`
  Output JSON file. Default: `rates.json`.
- `--temp`, `--kt`, `--beta`
  Mutually exclusive ways to specify temperature.
- `--timeunit`
  Conversion factor from the time unit in the COLVAR file to seconds.
- `--energyunit`
  Conversion factor from the energy unit in the COLVAR file to kJ/mol.
- `--tcol`, `--vcol`
  Time and bias column indices.
- `--acol`
  Acceleration-factor column index if present.
- `--mcol`
  Max-bias column index if present.
- `--logfiles`, `--maxlen`, `--maxtime`, `--numevents`
  Ways to determine which runs actually transitioned. Use exactly one when not all trajectories transition.
- `-m`, `-M`, `-k`, `-K`, `-e`, `-E`
  Select the estimator(s) to run.
- `-b`, `--bootstrap`
  Enable bootstrap uncertainty analysis.
- `-q`, `--quiet`
  Suppress terminal printing and only write JSON output.

Notes:

- `KTR` and regular `EATR` are not intended for OPES flooding trajectories. Use `eatr-flooding-analysis` for those.
- If your COLVAR file includes an acceleration column, passing `--acol` is preferable.
- If your COLVAR files were written in femtoseconds and you want SI rates, use `--timeunit 1e-15`.

### `eatr-flooding-analysis`

This command analyzes multiple sets of trajectories collected under different bias strengths and estimates a single unbiased `k0` plus a single `γ`.

Typical usage:

```bash
eatr-flooding-analysis \
  -i barrier5/*.colvar --barrier 5 \
  -i barrier10/*.colvar --barrier 10 \
  -i barrier15/*.colvar --barrier 15 \
  --temp 310
```

Equivalent form:

```bash
eatr-flooding-analysis \
  -i barrier5/*.colvar \
  -i barrier10/*.colvar \
  -i barrier15/*.colvar \
  --barriers 5 10 15 \
  --temp 310
```

Important arguments:

- `-i`, `--input`
  Supply one group of trajectory files per simulation set.
- `--barrier` or `--barriers`
  Bias-strength labels for each set. For OPES, this should usually be the PLUMED `BARRIER` value.
- `--timeunit`, `--energyunit`, `--temp`, `--kt`, `--beta`
  Unit and temperature handling, as in `eatr-analysis`.
- `--tcol`, `--vcol`, `--acol`
  Time, bias, and optional acceleration columns.
- `--logfiles`, `--maxlen`, `--maxtime`, `--numevents`
  Set-wise transition detection.
- `--cdf`
  Fit the observed rate for each set using the CDF instead of the MLE.
- `--timefirst`
  Change how the exponential average is aggregated.
- `--nooffset`
  Disable automatic addition of the OPES barrier offset to the reported bias.
- `--opesf`
  Also report the standard OPES-flooding estimate alongside EATR-flooding.

Notes:

- For OPES data produced with `OPES_METAD ... BARRIER=...`, you usually want to pass the same `BARRIER` values here and leave `--nooffset` unset.
- The method is also useful for MetaD if you have several sets with systematically varied deposition pace.

### `eatr-check-order`

This helper writes the expanded order of COLVAR files, optionally paired with log files, so you can verify shell glob expansion and pairing.

Example:

```bash
eatr-check-order -i run_*/metad.colvar -l run_*/p.log -o order.dat
```

## Example Data

The repository includes two example collections under [example-data/Ree_Data](/Volumes/HockyExtraSpace/Dropbox/research/projects/NNP-EATR-data-analysis/EATR-new-rate-scripts/example-data/Ree_Data):

- [E_end_end_distance_opes](/Volumes/HockyExtraSpace/Dropbox/research/projects/NNP-EATR-data-analysis/EATR-new-rate-scripts/example-data/Ree_Data/E_end_end_distance_opes)
  OPES flooding simulations with sets `eruns_barr5`, `7`, `9`, `11`, `13`
- [E_end_end_distance_wt](/Volumes/HockyExtraSpace/Dropbox/research/projects/NNP-EATR-data-analysis/EATR-new-rate-scripts/example-data/Ree_Data/E_end_end_distance_wt)
  WT-MetaD simulations with sets `eruns_pace1e2`, `1e3`, `1e4`, `2e4`, `5e4`, `1e5`, `5e5`, `1e6`

For these protein G examples, the LAMMPS inputs use `real` units with a `10 fs` timestep, so the correct time conversion is:

```bash
--timeunit 1e-15
```

The temperature used in the examples is:

```bash
--temp 312
```

## Worked Commands

### 1. OPES flooding example

To analyze the OPES datasets with EATR-flooding:

```bash
eatr-flooding-analysis \
  -i example-data/Ree_Data/E_end_end_distance_opes/eruns_barr5/run_*/opes_short.colvar --barrier 5 \
  -i example-data/Ree_Data/E_end_end_distance_opes/eruns_barr7/run_*/opes_short.colvar --barrier 7 \
  -i example-data/Ree_Data/E_end_end_distance_opes/eruns_barr9/run_*/opes_short.colvar --barrier 9 \
  -i example-data/Ree_Data/E_end_end_distance_opes/eruns_barr11/run_*/opes_short.colvar --barrier 11 \
  -i example-data/Ree_Data/E_end_end_distance_opes/eruns_barr13/run_*/opes_short.colvar --barrier 13 \
  --logfiles example-data/Ree_Data/E_end_end_distance_opes/eruns_barr5/run_*/p.log \
  --logfiles example-data/Ree_Data/E_end_end_distance_opes/eruns_barr7/run_*/p.log \
  --logfiles example-data/Ree_Data/E_end_end_distance_opes/eruns_barr9/run_*/p.log \
  --logfiles example-data/Ree_Data/E_end_end_distance_opes/eruns_barr11/run_*/p.log \
  --logfiles example-data/Ree_Data/E_end_end_distance_opes/eruns_barr13/run_*/p.log \
  --temp 312 \
  --timeunit 1e-15 \
  --tcol 0 \
  --vcol 4 \
  --opesf
```

Why these columns:

- in `opes_short.colvar`, column 0 is time
- column 4 is `opes.bias`

### 2. Regular EATR on one WT-MetaD set

For a single MetaD set such as `eruns_pace1e4`:

```bash
eatr-analysis \
  -i example-data/Ree_Data/E_end_end_distance_wt/eruns_pace1e4/run_*/metad.colvar \
  --logfiles example-data/Ree_Data/E_end_end_distance_wt/eruns_pace1e4/run_*/p.log \
  --temp 312 \
  --timeunit 1e-15 \
  --tcol 0 \
  --vcol 2 \
  --acol 4 \
  -eE \
  -o example-data/test_results/pace1e4_rates.json
```

Why these columns:

- in `metad.colvar`, column 0 is time
- column 2 is `metad.bias`
- column 4 is `metad.acc`

### 3. Flooding-style analysis across WT-MetaD pace sets

The flooding paper shows that EATR-flooding can also be applied to MetaD by comparing sets with different deposition pace. In that interpretation, the pace is the stepped biasing condition.

The command-line interface currently expects each set to be labeled by a numeric `--barrier` value. For MetaD pace ladders, the physically meaningful stepped variable is pace, not an OPES barrier, so the most reproducible workflow in this repository is the scripted example runner:

```bash
.venv/bin/python scripts/run_example_analyses.py
```

That script writes bootstrap-backed summaries and plots with reported rates converted to `us^-1` and pace units in `ps`. The current example workflow uses `50` trajectory-resampling bootstrap replicas per analysis. The generated files are:

- [wt_regular_eatr_summary.json](/Volumes/HockyExtraSpace/Dropbox/research/projects/NNP-EATR-data-analysis/EATR-new-rate-scripts/example-data/test_results/wt_regular_eatr_summary.json)
- [wt_regular_eatr_vs_pace.png](/Volumes/HockyExtraSpace/Dropbox/research/projects/NNP-EATR-data-analysis/EATR-new-rate-scripts/example-data/test_results/wt_regular_eatr_vs_pace.png)
- [wt_flooding_summary.json](/Volumes/HockyExtraSpace/Dropbox/research/projects/NNP-EATR-data-analysis/EATR-new-rate-scripts/example-data/test_results/wt_flooding_summary.json)
- [wt_flooding_all_paces.png](/Volumes/HockyExtraSpace/Dropbox/research/projects/NNP-EATR-data-analysis/EATR-new-rate-scripts/example-data/test_results/wt_flooding_all_paces.png)
- [wt_flooding_filtered_paces.png](/Volumes/HockyExtraSpace/Dropbox/research/projects/NNP-EATR-data-analysis/EATR-new-rate-scripts/example-data/test_results/wt_flooding_filtered_paces.png)
- [wt_observed_rate_vs_pace.png](/Volumes/HockyExtraSpace/Dropbox/research/projects/NNP-EATR-data-analysis/EATR-new-rate-scripts/example-data/test_results/wt_observed_rate_vs_pace.png)
- [opes_flooding_summary.json](/Volumes/HockyExtraSpace/Dropbox/research/projects/NNP-EATR-data-analysis/EATR-new-rate-scripts/example-data/test_results/opes_flooding_summary.json)
- [opes_flooding_diagnostics.png](/Volumes/HockyExtraSpace/Dropbox/research/projects/NNP-EATR-data-analysis/EATR-new-rate-scripts/example-data/test_results/opes_flooding_diagnostics.png)
- [opes_observed_rate_vs_barrier.png](/Volumes/HockyExtraSpace/Dropbox/research/projects/NNP-EATR-data-analysis/EATR-new-rate-scripts/example-data/test_results/opes_observed_rate_vs_barrier.png)
- [opes_ln_kobs_vs_acceleration.png](/Volumes/HockyExtraSpace/Dropbox/research/projects/NNP-EATR-data-analysis/EATR-new-rate-scripts/example-data/test_results/opes_ln_kobs_vs_acceleration.png)

## Example Results Produced In This Repository

The example workflow currently reports:

- OPES flooding fit:
  - `gamma_best ≈ 0.4204`
  - `logk0_best ≈ 20.9100`
- WT flooding fit using all pace sets:
  - `gamma_best ≈ 0.3641`
  - `logk0_best ≈ 20.6906`
- WT flooding fit using a filtered subset with pace `>= 100 ps`:
  - `gamma_best ≈ 0.2689`
  - `logk0_best ≈ 21.2035`

One MetaD set, `eruns_pace1e3`, produced non-finite values in the regular EATR example workflow, and this is recorded explicitly in the example summary JSON.

## Python Usage

The library functions remain available from Python, and the packaged CLI modules now separate the numerical analysis from output formatting:

- [eatr_rates/rates_cmd.py](/Volumes/HockyExtraSpace/Dropbox/research/projects/NNP-EATR-data-analysis/EATR-new-rate-scripts/eatr_rates/rates_cmd.py)
- [eatr_rates/rates_eatr_opes.py](/Volumes/HockyExtraSpace/Dropbox/research/projects/NNP-EATR-data-analysis/EATR-new-rate-scripts/eatr_rates/rates_eatr_opes.py)
- [rate_methods_library.py](/Volumes/HockyExtraSpace/Dropbox/research/projects/NNP-EATR-data-analysis/EATR-new-rate-scripts/rate_methods_library.py)

If you want to build automated regression tests, the easiest target is the example runner in [scripts/run_example_analyses.py](/Volumes/HockyExtraSpace/Dropbox/research/projects/NNP-EATR-data-analysis/EATR-new-rate-scripts/scripts/run_example_analyses.py) and the JSON outputs it writes.

## Tests

Run the unit tests with:

```bash
pytest
```

Or with the standard library test runner:

```bash
python3 -m unittest discover -s tests -v
```
