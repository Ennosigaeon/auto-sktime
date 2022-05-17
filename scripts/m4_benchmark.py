from argparse import ArgumentParser

from matplotlib import pyplot as plt

from autosktime.automl import AutoML
from autosktime.data.benchmark.m4 import load_timeseries, naive_2
from autosktime.metrics import OverallWeightedAverage
from sktime.utils.plotting import plot_series

parser = ArgumentParser()
parser.add_argument('dataset', type=str, help='Name of the M4 dataset')
parser.add_argument('--time_left_for_task', type=int, default=30, help='Optimization duration')
parser.add_argument('--time_per_run', type=int, default=10, help='Timeout for evaluating a single configuration')

args = parser.parse_args()

y_train, y_test = load_timeseries(args.dataset)
y_naive2 = naive_2(y_train)

automl = AutoML(
    time_left_for_this_task=args.time_left_for_task,
    per_run_time_limit=args.time_per_run
)
automl.fit(y_train, dataset_name=args.dataset)
y_pred = automl.predict(y_test.index)

owa = OverallWeightedAverage()
print('OWA', owa(y_test, y_pred, y_train=y_train))

fig, ax = plot_series(
    y_train, y_test, y_pred, y_naive2,
    labels=['y_train', 'y_test', 'y_pred', 'y_naive2'],
    y_label=args.dataset
)
plt.show()
