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
        'cmapss': [8.638309, 9.123938, 8.956812, 9.307966, 9.723507, 8.179532, 8.999519, 9.756667, 9.232296,
                   10.170651],
        'cmapss_2': [14.790167, 14.856804, 14.782307, 14.965282, 13.646124, 13.873828, 14.873185, 14.529404, 14.701387,
                     14.450477],
        'cmapss_3': [8.458211, 8.545085, 7.373021, 7.923177, 8.019972, 8.375418, 8.305244, 7.667346, 8.410680,
                     8.346721],
        'cmapss_4': [12.785706, 12.601181, 12.668047, 12.745517, 12.290330, 12.416788, 12.596339, 12.650204, 12.485093,
                     12.537129],
        'phme20': [7.06187, 7.11638, 16.727935, 6.531607, 6.18742, 5.840677, 19.591415, 6.298159, 7.489337, 6.413812],
        'femto_bearing': [16.742463, 17.270053, 28.83302, 14.529346, 21.113618, 12.366536, 32.175948, 19.347928,
                          36.444315, 26.389449],
        'ppm': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }
    print('autorul\n', pd.DataFrame(autorul).describe())

    # Kuersat2020 (baseline_rf)
    random_forest = {
        'cmapss': [14.348925, 14.390021, 14.349847, 14.416551, 14.347776, 14.390847, 14.423398, 14.405857, 14.437454,
                   14.369050],
        'cmapss_2': [18.857875, 18.720073, 18.887413, 18.762640, 18.780921, 18.812348, 18.844314, 18.785081, 18.832414,
                     18.761160],
        'cmapss_3': [12.410642, 12.370485, 12.394060, 12.416969, 12.400340, 12.377720, 12.450673, 12.394603, 12.467157,
                     12.415838],
        'cmapss_4': [13.657715, 13.666616, 13.721388, 13.676149, 13.710827, 13.618531, 13.666988, 13.651533, 13.630094,
                     13.661874],
        'phme20': [12.720683, 12.720683, 23.956588, 16.716901, 12.831065, 12.662925, 23.365915, 16.612625, 12.821174,
                   12.738881],
        'femto_bearing': [26.602833, 30.958374, 24.591263, 24.957461, 28.741304, 29.542713, 27.570313, 28.961271,
                          25.997710, 24.945848],
        'ppm': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }
    print('Kuersat2020\n', pd.DataFrame(random_forest).describe())

    # Zheng2017 (baseline_lstm)
    random_forest = {
        'cmapss': [14.628966, 14.832887, 14.906107, 16.035446, 14.210275, 14.488895, 14.484894, 14.782675, 14.092878,
                   15.357895],
        'cmapss_2': [17.766856, 18.411374, 16.634752, 17.413341, 19.367612, 17.504524, 18.053524, 18.098999, 17.968152,
                     18.563029],
        'cmapss_3': [11.765872, 12.145545, 11.972510, 12.236323, 11.701966, 11.676044, 21.784978, 11.083439, 11.425196,
                     11.787479],
        'cmapss_4': [11.092113, 12.053612, 11.773727, 11.563602, 11.534442, 11.383793, 11.384072, 11.754583, 12.206736,
                     11.837478],
        'phme20': [13.331706, 14.387351, 24.267973, 17.955105, 14.062440, 14.754616, 23.876763, 17.953770, 13.686663,
                   13.950022],
        'femto_bearing': [35.682707, 29.855155, 34.665040, 34.079405, 37.834681, 42.547187, 40.539804, 34.938149,
                          26.701164, 38.956260],
        'ppm': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }
    print('Zheng2017\n', pd.DataFrame(random_forest).describe())

    # Li2018b (baseline_cnn)
    random_forest = {
        'cmapss': [17.729213, 16.880596, 16.124352, 15.996100, 16.970271, 17.615540, 17.599030, 15.871664, 16.977988,
                   16.800957],
        'cmapss_2': [21.162352, 24.103969, 23.059695, 23.370648, 23.680346, 23.755606, 23.960246, 23.644329, 24.807948,
                     25.320151],
        'cmapss_3': [13.944223, 14.551544, 13.535131, 13.201010, 13.743129, 13.623395, 14.306888, 13.619146, 13.324463,
                     14.008844],
        'cmapss_4': [18.994669, 18.510300, 19.392269, 18.153070, 18.156013, 18.143621, 18.494288, 18.108049, 19.799347,
                     19.902652],
        'phme20': [22.156817, 21.697822, 30.472827, 21.232256, 22.879465, 21.960380, 29.225613, 24.911734, 19.786796,
                   21.394286],
        'femto_bearing': [31.481675, 36.516447, 33.067918, 33.399513, 31.853231, 35.025026, 31.081165, 30.416512,
                          26.887195, 31.578534],
        'ppm': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }
    print('Li2018b\n', pd.DataFrame(random_forest).describe())


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
