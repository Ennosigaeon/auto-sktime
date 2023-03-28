# auto-sktime

Automatic creation of time series forecasts, regression and classification.

## Installation

For trouble shooting and detailed installation instructions, see the documentation.

```
Operating system: Linux
Python version: Python 3.8, 3.9, and 3.10 (only 64 bit)
Package managers: pip
```

### pip

auto-sktime is available in pip. You can see all available wheels [here](https://test.pypi.org/project/auto-sktime).

```bash
pip install auto-sktime
```

or, with maximum dependencies,

```bash
pip install auto-sktime[all_extras]
```

## Remaining Useful Life Predictions (AutoRUL)

This section describes how to reproduce the results in the _AutoRUL_ paper. First, checkout the exact code that was used
to create the results. Therefore, you can use the tag [v0.1.0](https://github.com/Ennosigaeon/auto-sktime/tree/v0.1.0)

```bash
git checkout tags/v0.1.0 -b autorul
```

Next, switch to the `scripts` directory and use

```bash
python remaining_useful_lifetime.py <BENCHMARK>
```

to run a single benchmark data set. To view the available benchmarks and all configuration parameters run

```bash
python remaining_useful_lifetime.py --help
```

### Reproducing results

You can use the following commands to recreate the reported baseline results in the experiments of the paper.

```bash
python remaining_useful_lifetime.py <BENCHMARK> --runcount_limit 1 --timeout 3600 --multi_fidelity False --include baseline_lstm
python remaining_useful_lifetime.py <BENCHMARK> --runcount_limit 1 --timeout 3600 --multi_fidelity False --include baseline_cnn
python remaining_useful_lifetime.py <BENCHMARK> --runcount_limit 1 --timeout 3600 --multi_fidelity False --include baseline_transformer
python remaining_useful_lifetime.py <BENCHMARK> --runcount_limit 1 --timeout 7200 --multi_fidelity False --include baseline_rf
python remaining_useful_lifetime.py <BENCHMARK> --runcount_limit 200 --timeout 7200 --multi_fidelity False --ensemble_size 1 --include baseline_svm
```

with `<BENCHMARK>` being one of `{cmapss,cmapss_1,cmapss_2,cmapss_3,cmapss_4,femto_bearing,filtration,phm08,phme20}`.
For the _AutoRUL_ evaluation only the benchmark is provided and all remaining default configurations are used.

```bash
python remaining_useful_lifetime.py <BENCHMARK>
```

To reproduce the results from AutoCoevoRUL, checkout the [repository](https://github.com/Ennosigaeon/AutoCoevoRUL) from
Github and use the [autocoevorul.py](scripts/autocoevorul.py) file to either export the data sets or import the results.

## Note

This project has been set up using PyScaffold 4.2.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.

## Building

To create a new release of `auto-sktime` you will have to install `build` and `twine`

```bash
pip install build twine
python -m build

```
