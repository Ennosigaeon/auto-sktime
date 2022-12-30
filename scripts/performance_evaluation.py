import numpy as np
import os

import pickle

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from smac.runhistory.runhistory import RunHistory
from smac.tae import StatusType

from autosktime.data.benchmark import BENCHMARKS

pd.options.display.float_format = "{:,.2f}".format


def print_raw_performance():
    autorul = {
        'cmapss_1': [8.638309, 9.123938, 8.956812, 9.307966, 9.723507, 8.179532, 8.999519, 9.756667, 9.232296,
                     10.170651],
        'cmapss_2': [14.790167, 14.856804, 14.782307, 14.965282, 13.646124, 13.873828, 14.873185, 14.529404, 14.701387,
                     14.450477],
        'cmapss_3': [8.458211, 8.545085, 7.373021, 7.923177, 8.019972, 8.375418, 8.305244, 7.667346, 8.410680,
                     8.346721],
        'cmapss_4': [12.785706, 12.601181, 12.668047, 12.745517, 12.290330, 12.416788, 12.596339, 12.650204, 12.485093,
                     12.537129],
        'phme20': [7.06187, 7.11638, 6.727935, 6.531607, 6.18742, 5.840677, 9.591415, 6.298159, 7.489337, 6.413812],
        'femto': [16.742463, 17.270053, 28.83302, 14.529346, 21.113618, 12.366536, 32.175948, 19.347928, 36.444315,
                  26.389449],
        'ppm': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }
    print('autorul\n', pd.DataFrame(autorul).describe())

    # Kuersat2020
    random_forest = {
        'cmapss_1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'cmapss_2': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'cmapss_3': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'cmapss_4': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'phme20': [12.720683, 12.720683, 13.956588, 16.716901, 12.831065, 12.662925, 13.365915, 16.612625, 12.821174,
                   12.738881],
        'femto': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'ppm': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }
    print('Kuersat2020\n', pd.DataFrame(random_forest).describe())


def plot_any_time_performance():
    def cache_runhistories():
        runhistories = {}
        for _, benchmark in BENCHMARKS.items():
            benchmark = benchmark()
            runhistories[benchmark.name()] = []
            for fold in range(10):
                try:
                    # CMAPSS_3, fold 0 is missing
                    with open(f'results/final/{benchmark.name()}/fold_{fold}/model.pkl', 'rb') as f:
                        automl = pickle.load(f)
                        runhistories[benchmark.name()].append(automl.runhistory_)
                except Exception as ex:
                    print(benchmark.name(), fold, ex)
        with open('results/final/runhistory.pkl', 'wb') as f:
            pickle.dump(runhistories, f)

    def calculate_performance(rh: RunHistory):
        performance_over_time = []

        start = None
        for value in rh.data.values():
            if start is None:
                # Actual start time is not recorded. Use pessimistic guess for start up time based on log files
                start = value.starttime - 15

            cost = value.additional_info.get('test_loss', {'RootMeanSquaredError': value.cost})[
                'RootMeanSquaredError']
            if value.status == StatusType.SUCCESS and cost < 100:
                performance_over_time.append([value.endtime - start, cost, 0])

        performance_over_time = np.array(performance_over_time)
        best_performance = np.min(performance_over_time, axis=0)[1]
        performance_over_time[:, 0] = np.ceil(performance_over_time[:, 0])
        performance_over_time[:, 1] = performance_over_time[:, 1] - best_performance
        performance_over_time[:, 2] = np.minimum.accumulate(performance_over_time[:, 1]) * 2

        grid_performance = np.tile(np.arange(0, 36000, dtype=float), (3, 1)).T
        previous = None
        for row in performance_over_time:
            if previous is None:
                grid_performance[0:int(row[0]), 1:] = -1000
            else:
                grid_performance[previous:int(row[0]), 1:] = row[1:]
            previous = int(row[0])
        grid_performance[previous:36000, 1:] = row[1:]

        return grid_performance

    if not os.path.exists('results/final/runhistory.pkl'):
        cache_runhistories()

    with open('results/final/runhistory.pkl', 'rb') as f:
        runhistories = pickle.load(f)

    benchmark_performances = []
    for benchmark in runhistories.keys():
        if len(runhistories[benchmark]) == 0:
            continue

        performances_over_time = np.array([calculate_performance(rh) for rh in runhistories[benchmark]])
        performance_over_time = np.mean(performances_over_time, axis=0)
        performance_over_time[:, 1] = np.std(performances_over_time, axis=0)[:, -1]
        benchmark_performances.append(performance_over_time)

    valid = ~np.any(np.array(benchmark_performances) < 0, axis=0).any(axis=1)
    benchmark_performances = np.mean(benchmark_performances, axis=0)[valid]

    x = benchmark_performances[:, 0]
    mean = benchmark_performances[:, 2]
    std = benchmark_performances[:, 1]

    df = pd.DataFrame(benchmark_performances, columns=['x', 'ci', 'mean'])

    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    sns.lineplot(df, ax=ax, x='x', y='mean', errorbar='ci')
    ax.fill_between(x, mean + std, mean - std, facecolor='blue', alpha=0.2)

    plt.axvline(x=600, color='red', ls=':', label='10 min')
    plt.text(600, 7.1, ' 10 minutes')
    plt.axvline(x=3600, color='red', ls=':')
    plt.text(3600, 7.1, ' 1 hour')
    plt.axvline(x=18000, color='red', ls=':')
    plt.text(18000, 7.1, ' 5 hours')

    plt.xlabel('Wall Clock Time [Seconds]')
    plt.ylabel('Immediate Regret')
    # plt.yscale('log')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('results/final/performance-over-time.pdf')
    plt.show()


if __name__ == '__main__':
    print_raw_performance()
    plot_any_time_performance()