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
to create the results. Therefore, you can use the tag [TODO](https://github.com/Ennosigaeon/auto-sktime)

```bash
git checkout -t <TODO>
```

Next, switch to the `scripts` directory and use

```bash
python remaining_useful_lifetime.py <BENCHMARK>
```

to run a single benchmark data set. To view the available benchmarks and all configuration parameters run

```bash
python remaining_useful_lifetime.py --help
```

For the experiments in the paper, only the benchmark is provided and all remaining default configurations are used.

## Note

This project has been set up using PyScaffold 4.2.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.

## Building

To create a new release of `auto-sktime` you will have to install `build` and `twine`
```bash
pip install build twine
python -m build

```
