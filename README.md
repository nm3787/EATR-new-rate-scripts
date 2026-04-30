# Rate Analysis Scripts

## Installation

This repository can now be installed as a Python package:

```bash
pip install .
```

For development work, including the test dependencies:

```bash
pip install -e ".[dev]"
```

The package installs these command-line tools:

- `eatr-analysis`
- `eatr-flooding-analysis`
- `eatr-check-order`

You can also run the main CLI with `python -m eatr_rates`.

`rate_methods_library.py` is the Python library that contains all of the methods needed to perform all of the available analyses. It is imported by `rates_cmd.py` and `rates_eatr_opes.py`.

## Metadynamics Analyses and OPES-Flooding
`rates_cmd.py` is the Python script that can run the following analyses:

- Infrequent Metadynamics
  - Using the mean residence time estimator as in \[Tiwary and Parinello, Phys. Rev. Lett. 2013, 111, 230602.\]  ("iMetaD MLE")
  - Using least-squares fitting on the cumulative density function (CDF) as in \[Salvalaglio et al. J. Chem. Theory Comput. 2014, 10, 4, 1420-1425.\] ("iMetaD CDF")
- OPES-Flooding
  - As in \[Ray et al. J. Chem. Theory Comput. 2022, 18, 11, 6500-6509.\] (Use "iMetaD CDF" but also specify BARRIER)
- Kramers' Time-dependent Rate (KTR)
  - Using the maximum likelihood estimator as in \[Palacio-Rodriguez et al. J. Phys. Chem. Lett. 2022, 13, 32, 7490-7496.\] ("KTR MLE")
  - Using least-squares fitting on the CDF as in \[Mazzaferro et al. J. Chem. Theory Comput. 2024, 20, 14, 5901-5912.\] ("KTR CDF")
- Exponential Average Time-dependent Rate (EATR) as in \[Mazzaferro et al. J. Chem. Theory Comput. 2024, 20, 14, 5901-5912.\]
  - Using the maximum likelihood estimator. ("EATR MLE")
  - Using least-squares fitting on the CDF. ("EATR CDF")

`rates_cmd.py` should be run using a command similar to:
`python rates_cmd.py -i run_*/*.colvar --temp 310 -MEb`

After installation, the equivalent packaged command is:
`eatr-analysis -i run_*/*.colvar --temp 310 -MEb`

Use `python rates_cmd.py -h` to get a full list of useful parameters, but some facts that are important to know:

- You can only use one of TEMP, KT, and BETA to specify the temperature. If you use TEMP, you must ensure that ENERGYUNIT is correct. For example, if you specify TEMP, and you had used kcal/mol in PLUMED, you must set ENERGYUNIT to 4.184 kJ/mol.
- In practice, an incorrect value for TIMEUNIT only puts the final value for the unbiased rate in the wrong units.
- ACOL and MCOL are not necessary. The script will calculate the acceleration factor and max bias from the information in the COLVAR file, but this may be slightly less accurate, especially if you printed to the COLVAR file very infrequently.
- If not all of your simulations transitioned, you should specify exactly one of LOGFILES, MAXLEN, MAXTIME, and NUMEVENTS. The analyses in this script uses the number of incomplete simulations to gain more information about the rate. If you specify LOGFILES, you should make sure that the Unix globs for the COLVAR files and the log files expand in the same order. This can be done using the `check_order.py` script.
- Including the BOOTSTRAP flag activates bootstrapping as an error analysis. If your Python installation includes a version of SciPy with the bootstrap method, this script will use that and return a 95% confidence interval. If not, there is an internal implementation which returns the standard error.
- Including the BAYESOPT flag sets the optimizer to the BayesianOptimization method from the bayes\_opt module. This does not come with Anaconda, so install it if you wish to use it. It takes much longer, but it is a global optimizer, which can sometimes help give reasonable results in KTR and EATR CDF.
- Including the LOGTRICK flag causes the script to use the "log-sum-exp" trick to add together (or in this case, integrate) exponentials of very large numbers with good precision. I personally have not found an instance where it is useful, but if you need it, it is available.
- You should NOT apply the KTR and EATR methods on simulations that were biased using OPES. You should use `rates_eatr_opes.py` instead.


## EATR-OPES
`rates_eatr_opes.py` is a Python script that can perform the EATR-OPES analysis on multiple sets of OPES simulations, where each set is given a different amount of bias.

`rates_eatr_opes.py` should be run using a command similar to: `python rates_eatr_opes.py -i barrier5/*.colvar --barrier 5 -i barrier10/*.colvar --barrier 10 -i barrier15/*.colvar --barrier 15 --temp 310` or `python rates_eatr_opes.py -i barrier5/*.colvar -i barrier10/*.colvar -i barrier15/*.colvar --barriers 5 10 15 --temp 310`

After installation, the equivalent packaged command is `eatr-flooding-analysis`.

Use `python rates_eatr_opes.py -h` to get a full list of useful parameters, but some facts that are important to know:

- You need to specify INPUT once for every set of simulations you performed.
- You also need to specify the value of the barrier parameter used in PLUMED for each set of simulations. You can either specify BARRIER multiple times, in the same order as you specify the sets of COLVAR files in INPUT, or you can specify BARRIERS with multiple arguments at once.
- If you want to specify LOGFILES, that should also be specified for each set of simulations.
- Bootstrapping in this script can only use an internal implementation, and thus will return the standard error. The samples are drawn by sampling with replacement the simulations within each set separately.
- Because this script gives a single result, the result does not get saved to a file, but is instead only printed to the terminal.

## Tests

Run the unit test suite with:

```bash
pytest
```
