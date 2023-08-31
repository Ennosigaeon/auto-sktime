import os

import numpy as np
import pandas as pd
from critdd import Diagram

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 10000)
pd.set_option("display.precision", 4)


def analyse_results(with_timeout: bool = True, with_missing_values: bool = True):
    print(f'with_timeout = {with_timeout}, with_missing_values = {with_missing_values}')

    for time in (300, 60):
        benchmark = {
            'auto-pytorch': pd.read_csv(f'results/auto-pytorch-{time}.csv'),
            'auto-sktime': pd.read_csv(f'results/auto-sktime-{time}.csv'),
            'autogluon': pd.read_csv('results/autogluon.csv'),
            # 'autogluon_hpo': pd.read_csv('results/autogluon.hpo.csv'),
            # 'autots_default': pd.read_csv(f'results/autots-{time}.default.csv'),
            'autots': pd.read_csv(f'results/autots-{time}.csv'),
            'hyperts': pd.read_csv(f'results/hyperts-{time}.csv'),
            'pmdarima': pd.read_csv('results/pmdarima.csv'),
            'prophet': pd.read_csv('results/prophet.csv'),
            'pyaf': pd.read_csv('results/pyaf.csv')
        }
        ablation_study = {
            'auto-sktime': pd.read_csv(f'results/auto-sktime-{time}.csv'),
            'auto-sktime_templates': pd.read_csv(f'results/auto-sktime_templates-{time}.csv'),
            'auto-sktime_multi_fidelity': pd.read_csv(f'results/auto-sktime_multi_fidelity-{time}.csv'),
            'auto-sktime_warm_starting': pd.read_csv(f'results/auto-sktime_warm_starting-{time}.csv'),
            'auto-pytorch': pd.read_csv(f'results/auto-pytorch-{time}.csv'),
        }

        for results in (ablation_study, benchmark):
            print(f'###############\n'
                  f'Benchmark {results == benchmark}, Duration {time}, Timeout {with_timeout}, Missing Values {with_missing_values}\n'
                  f'###############\n')

            df = pd.concat(results.values())

            if not with_missing_values:
                nan = df[(df['method'] == 'pmdarima') & np.isinf(df['smape'])]['dataset'].unique()
                df = df[~df['dataset'].isin(nan)]

            if with_timeout:
                df.loc[(df['duration'] > time + 60), 'smape'] = np.inf
            df = df.drop(columns=['seed']).replace([np.inf], 2).round(5)

            print('Number of timeouts')
            print(df.loc[(df['duration'] > time + 60), 'method'].value_counts())

            mean = df.groupby(['dataset', 'method']).mean(numeric_only=True).reset_index()
            std = df.groupby(['dataset', 'method']).std(numeric_only=True).reset_index()

            mean['performance_rank'] = mean.groupby('dataset').rank()['smape']
            mean['duration_rank'] = mean.groupby('dataset').rank()['duration']
            mean['smape_std'] = std['smape']
            mean['duration_std'] = std['duration']

            print(mean.groupby('method').mean(numeric_only=True), end='\n\n\n')

            wide = mean[['dataset', 'method', 'smape']].pivot(index='dataset', columns='method', values='smape')
            diagram = Diagram(
                wide.to_numpy(),
                treatment_names=wide.columns,
                maximize_outcome=False
            )
            filename = f'cd_{"benchmark" if results == benchmark else "ablation"}-{time}{"_timeout" if with_timeout else ""}{"_nan" if with_missing_values else ""}.pdf'
            diagram.to_file(filename, alpha=.05, adjustment='holm', reverse_x=False)
            for suffix in ('.aux', '.log', '.tex'):
                os.remove(filename.replace('.pdf', suffix))


analyse_results(with_timeout=True, with_missing_values=True)
analyse_results(with_timeout=False, with_missing_values=True)
analyse_results(with_timeout=True, with_missing_values=False)
analyse_results(with_timeout=False, with_missing_values=False)
