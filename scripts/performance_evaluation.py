import json
import os
import pickle

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import wilcoxon
from scipy.stats._morestats import WilcoxonResult
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.enumerations import StatusType

from autosktime.data.benchmark import BENCHMARKS

pd.set_option('display.float_format', '{:,.2f}'.format)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def print_raw_performance():
    autorul = {
        'FD001': [8.638309, 9.123938, 8.956812, 9.307966, 9.723507, 8.179532, 8.999519, 9.756667, 9.232296,
                  10.170651],
        'FD002': [14.790167, 14.856804, 14.782307, 14.965282, 13.646124, 13.873828, 14.873185, 14.529404, 14.701387,
                  14.450477],
        'FD003': [8.458211, 8.545085, 7.373021, 7.923177, 8.019972, 8.375418, 8.305244, 7.667346, 8.410680,
                  8.346721],
        'FD004': [12.785706, 12.601181, 12.668047, 12.745517, 12.290330, 12.416788, 12.596339, 12.650204, 12.485093,
                  12.537129],
        'PHME\'20': [7.06187, 7.11638, 16.727935, 6.531607, 6.18742, 5.840677, 19.591415, 6.298159, 7.489337, 6.413812],
        'PRONOSTIA': [26.973312, 27.396291, 28.83302, 14.529346, 21.113618, 12.366536, 22.175948, 19.347928,
                      26.087229, 26.389449],
        'Filtration': [4.97182, 5.101947, 6.522199, 6.799997, 5.759050, 5.114282, 6.792338, 5.266525, 5.165513,
                       6.990814],
        'PHM\'08': [26.6776846821459, 26.5677479286446, 27.6308342255532, 29.8473734187784, 29.5025851409669,
                    27.3521028442056, 28.0737740605, 26.8072160061428, 26.7306591763091, 28.3979229874299],
    }
    print('autorul\n', pd.DataFrame(autorul).describe())

    # Kuersat2020 (baseline_rf)
    random_forest = {
        'FD001': [14.348925, 14.390021, 14.349847, 14.416551, 14.347776, 14.390847, 14.423398, 14.405857, 14.437454,
                  14.369050],
        'FD002': [18.857875, 18.720073, 18.887413, 18.762640, 18.780921, 18.812348, 18.844314, 18.785081, 18.832414,
                  18.761160],
        'FD003': [12.410642, 12.370485, 12.394060, 12.416969, 12.400340, 12.377720, 12.450673, 12.394603, 12.467157,
                  12.415838],
        'FD004': [13.657715, 13.666616, 13.721388, 13.676149, 13.710827, 13.618531, 13.666988, 13.651533, 13.630094,
                  13.661874],
        'PHME\'20': [12.720683, 12.720683, 23.956588, 16.716901, 12.831065, 12.662925, 23.365915, 16.612625, 12.821174,
                     12.738881],
        'PRONOSTIA': [26.602833, 30.958374, 24.591263, 24.957461, 28.741304, 29.542713, 27.570313, 28.961271,
                      25.997710, 24.945848],
        'Filtration': [6.427489, 5.574944, 6.955994, 7.228762, 6.305010, 5.943331, 7.449933, 6.893373, 6.246812,
                       7.870088],
        'PHM\'08': [31.2410556799862, 31.139835741378, 30.4387481674263, 31.1912734590943, 30.6047774048432,
                    30.9386972899636, 31.4550512636683, 31.5081027673835, 30.8049513877234, 30.7838623307732],
    }
    print('Kuersat2020\n', pd.DataFrame(random_forest).describe())

    # Zheng2017 (baseline_lstm)
    lstm = {
        'FD001': [14.628966, 14.832887, 14.906107, 16.035446, 14.210275, 14.488895, 14.484894, 14.782675, 14.092878,
                  15.357895],
        'FD002': [17.766856, 18.411374, 16.634752, 17.413341, 19.367612, 17.504524, 18.053524, 18.098999, 17.968152,
                  18.563029],
        'FD003': [11.765872, 12.145545, 11.972510, 12.236323, 11.701966, 11.676044, 11.796663, 11.083439, 11.425196,
                  11.787479],
        'FD004': [11.092113, 12.053612, 11.773727, 11.563602, 11.534442, 11.383793, 11.384072, 11.754583, 12.206736,
                  11.837478],
        'PHME\'20': [13.331706, 14.387351, 24.267973, 17.955105, 14.062440, 14.754616, 23.876763, 17.953770, 13.686663,
                     13.950022],
        'PRONOSTIA': [35.682707, 29.855155, 34.665040, 34.079405, 37.834681, 42.547187, 40.539804, 34.938149,
                      26.701164, 38.956260],
        'Filtration': [5.545543, 5.693300, 6.997323, 7.183846, 5.843927, 6.034569, 7.196829, 6.357253, 5.390593,
                       6.954364],
        'PHM\'08': [29.0272928465608, 28.9718209645165, 27.8433088910065, 27.5929329720492, 28.2991393862075,
                    28.7816448452829, 27.7867022152684, 26.7472224726232, 29.8628255528508, 28.3009296667088],
    }
    print('Zheng2017\n', pd.DataFrame(lstm).describe())

    # Li2018b (baseline_cnn)
    cnn = {
        'FD001': [17.729213, 16.880596, 16.124352, 15.996100, 16.970271, 17.615540, 17.599030, 15.871664, 16.977988,
                  16.800957],
        'FD002': [23.089831, 23.742979, 23.059695, 23.370648, 23.680346, 23.755606, 23.960246, 23.644329, 23.991244,
                  23.369883],
        'FD003': [13.944223, 14.551544, 13.535131, 13.201010, 13.743129, 13.623395, 14.306888, 13.619146, 13.324463,
                  14.008844],
        'FD004': [18.994669, 18.510300, 19.392269, 18.153070, 18.156013, 18.143621, 18.494288, 18.108049, 19.799347,
                  19.902652],
        'PHME\'20': [22.156817, 21.697822, 30.472827, 21.232256, 22.879465, 21.960380, 29.225613, 24.911734, 19.786796,
                     21.394286],
        'PRONOSTIA': [31.481675, 36.516447, 33.067918, 33.399513, 31.853231, 35.025026, 31.081165, 30.416512,
                      26.887195, 31.578534],
        'Filtration': [14.846346, 15.259048, 15.637210, 15.423385, 14.842429, 12.992488, 15.865576, 15.181291,
                       14.143920, 15.784535],
        'PHM\'08': [31.8637395796539, 31.2967496714914, 31.3157188964264, 31.7201670865713, 30.8540238866829,
                    31.3713429741221, 31.3097147863087, 31.7425928367548, 31.5043430339374, 31.74815427706],
    }
    print('Li2018b\n', pd.DataFrame(cnn).describe())

    # Mo2021a (baseline_transformers)
    transformers = {
        'FD001': [14.218519, 14.242915, 14.136745, 13.917528, 14.255089, 14.195726, 12.368505, 14.440638, 13.037204,
                  13.679714],
        'FD002': [14.394549, 14.297733, 14.336771, 14.660137, 14.432021, 14.418107, 14.896199, 14.450827, 14.451664,
                  14.296273],
        'FD003': [12.767812, 12.916057, 12.870062, 12.722813, 12.899314, 13.020205, 12.670074, 12.791419, 12.996298,
                  12.905590],
        'FD004': [12.424973, 12.462107, 12.479720, 12.476005, 12.334822, 12.512587, 13.017701, 12.618935, 12.288414,
                  12.368177],
        'PHME\'20': [10.373761, 9.194543, 13.047224, 13.179986, 10.349902, 9.143935, 12.949856, 13.167748, 10.222071,
                     9.245085],
        'PRONOSTIA': [52.552287, 50.596645, 50.506670, 53.079521, 48.841844, 53.347745, 49.386437, 51.815780,
                      48.543660, 51.833493],
        'Filtration': [6.150689, 7.518467, 6.987488, 7.040194, 6.523790, 7.100308, 7.391409, 6.517309, 7.351363,
                       7.128781],
        'PHM\'08': [30.1522053919776, 29.9894071298517, 30.4635752990354, 29.167937019954, 30.1972975612057,
                    29.9163329972107, 30.0778942082055, 30.7615449872077, 29.3262740558701, 29.1787007935583],
    }
    print('Mo2021a\n', pd.DataFrame(transformers).describe())

    # Nieto2015 (baseline_svm)
    svm = {
        'FD001': [13.985249, 13.643011, 14.326570, 13.685853, 14.155676, 13.971911, 14.203493, 13.972439, 13.865017,
                  13.511654],
        'FD002': [20.165094, 19.841232, 19.820924, 20.467202, 19.799738, 21.329561, 20.410972, 20.221130, 19.850274,
                  19.912705],
        'FD003': [12.849954, 12.516817, 12.462559, 12.174377, 13.111373, 12.813108, 12.368642, 12.539270, 12.667587,
                  13.111124],
        'FD004': [16.302016, 16.589758, 15.792411, 16.053102, 15.013072, 16.020558, 16.415584, 16.750024, 16.228248,
                  16.061023],
        'PHME\'20': [27.063639, 27.865125, 33.608650, 31.299890, 26.948634, 27.781249, 33.523778, 31.429802, 27.378507,
                     27.806960],
        'PRONOSTIA': [27.031654, 30.327764, 25.598859, 27.326580, 28.801493, 29.076716, 28.621050, 28.950722,
                      27.618004, 25.663182],
        'Filtration': [8.615825, 8.395696, 9.049813, 9.845869, 9.50282, 8.663529, 9.547853, 8.468463, 7.823363,
                       10.124959],
        'PHM\'08': [44.7784334250317, 46.1944087525752, 46.1944087525752, 46.1944087525752, 45.6838242707416,
                    34.157903916956, 31.645400929677, 33.0865259584623, 35.1994147678623, 43.4222857528251],
    }
    print('Nieto2015\n', pd.DataFrame(svm).describe())

    autocoevorul = {
        'FD001': [16.259029433148285, 14.839146727405224, 14.782005913316578, 15.587311011260736, 14.574158361153689,
                  15.504548956605554, 15.500938333856695, 15.62067105928699, 16.703988056775746, 16.67932951694009],
        'FD002': [18.540305556341014, 16.79569806173818, 18.293760922174776, 16.61590882038273, 18.70850061150374,
                  18.399885213850972, 16.683376026628256, 16.18120818010116, 18.736131843067934, 18.92463186765048],
        'FD003': [16.16792967838529, 15.437392290060638, 16.12887429502924, 15.125943678212986, 14.829162058571958,
                  16.19167973117422, 13.999458002206236, 16.50542177378143, 16.809375367523824, 15.011441352716783],
        'FD004': [16.938912794800117, 17.967638284895678, 16.603693429809105, 15.506281718098252, 15.741299378638873,
                  16.817718076636464, 16.643901157032094, 15.83155859288426, 14.69420236616984, 16.63209241013984],
        'PHME\'20': [14.635979284249483, 16.02044229666742, 16.398019864653072, 15.193066260438986, 11.044510529891704,
                     13.635713786418487, 18.92627219772393, 18.868193702728215, 12.753430935608408, 14.822842886594522],
        'PRONOSTIA': [37.2491796984193, 24.41982910253519, 32.64477679769104, 27.561730709031558, 40.98508925090833,
                      29.075711100021962, 34.682691731072026, 42.78463529659524, 30.712367371657535,
                      30.611359185854138],
        'Filtration': [10.987414467922517, 10.771928376792468, 11.398574867015204, 11.369101980371754,
                       11.11230021078754, 10.71441951988408, 10.524690991158407, 10.374772263119946, 11.475661418947184,
                       11.183602782536962],
        'PHM\'08': [45.18158142, 48.12007897, 46.54629416, 45.54495911, 45.97452773, 47.96650706, 44.7264452,
                    46.81558715,
                    45.32076787, 46.76404602],
    }
    print('Tornede2021\n', pd.DataFrame(autocoevorul).describe())

    results = {
        'LSTM': lstm,
        'CNN': cnn,
        'Transformers': transformers,
        'RF': random_forest,
        'SVM': svm,
        'AutoCoevoRUL': autocoevorul,
        'AutoRul': autorul
    }

    for dataset in svm.keys():
        best = min(results, key=lambda k: np.average(results[k][dataset]))
        equivalent = []
        for method in results.keys():
            if method == best:
                continue
            a = np.array(results[best][dataset])
            b = np.array(results[method][dataset])
            res: WilcoxonResult = wilcoxon(a, b)
            if res.pvalue > 0.05:
                equivalent.append((method, res.pvalue))
        print(dataset, best, equivalent)

    sns.set_style("darkgrid")
    for dataset in svm.keys():
        data = []

        for method in results.keys():
            data += [{'method': method, 'value': val} for val in results[method][dataset]]

        df = pd.DataFrame(data)

        fig, ax = plt.subplots(figsize=(6, 3))
        sns.boxplot(data=df, x="method", y="value", ax=ax)
        ax.set(xlabel=None, ylabel='RMSE', title=dataset)
        plt.tight_layout()
        plt.savefig(f'results/final/performance-{dataset}.pdf')
        plt.show()


def evaluate_runhistory_statistics():
    for benchmark in BENCHMARKS.keys():
        total = []
        success = []
        timeout = []
        failures = []
        full_budget = []

        for fold in range(10):
            with open(f'results/final/{benchmark}/fold_{fold}/smac3-output/run_{fold}/runhistory.json') as f:
                rh = json.load(f)['data']
            total.append(len(rh))
            success.append(len([e for e in rh if e[1][2]['__enum__'] == 'StatusType.SUCCESS']))
            timeout.append(len([e for e in rh if e[1][2]['__enum__'] == 'StatusType.TIMEOUT']))
            failures.append(total[-1] - success[-1] - timeout[-1])
            full_budget.append(len([e for e in rh if e[0][3] == 100.0]))
        print(benchmark, np.std(total), np.std(success), np.std(failures), np.std(timeout), np.std(full_budget))


def evaluate_generated_pipelines():
    def cache_models():
        models = {}
        for _, benchmark in BENCHMARKS.items():
            benchmark = benchmark()
            models[benchmark.name()] = []
            for fold in range(10):
                try:
                    # CMAPSS_3, fold 0 is missing
                    with open(f'results/final/{benchmark.name()}/fold_{fold}/model.pkl', 'rb') as f:
                        automl = pickle.load(f)
                        print(benchmark.name(), len(automl.models_))
                        models[benchmark.name()] += [model.config for _, model in automl.models_]
                except Exception as ex:
                    print(benchmark.name(), fold, ex)
        with open('results/final/models.pkl', 'wb') as f:
            pickle.dump(models, f)

    if not os.path.exists('results/final/models.pkl'):
        cache_models()

    with open('results/final/models.pkl', 'rb') as f:
        models = pickle.load(f)

    for key, value in models.items():
        configs = [config.get_dictionary() for config in value]
        algorithms = []
        for config in configs:
            algorithms += [f'{key}:{value}' for key, value in config.items() if '__choice__' in key or
                           'nn-panel-regression:reduction:estimator:network:rnn:cell_type' in key]
        df = pd.Series(algorithms)
        print(df.value_counts().sort_index() / len(configs), end='\n\n\n')


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
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
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
    evaluate_generated_pipelines()
    print_raw_performance()
    evaluate_runhistory_statistics()
    plot_any_time_performance()
