import os

import numpy as np
import pandas as pd
from critdd import Diagram
from scipy.stats import ttest_ind

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 10000)
pd.set_option("display.precision", 2)


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
            # 'prophet': pd.read_csv('results/prophet.csv'),
            'pyaf': pd.read_csv('results/pyaf.csv'),
            'tft': pd.read_csv(f'results/tft-{time}.csv')
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

            rank = df.groupby(['dataset', 'method']).mean(numeric_only=True).groupby('dataset').rank()[
                'smape'].reset_index().rename(columns={'smape': 'performance_rank'})
            df = pd.merge(df, rank, how='left', left_on=['method', 'dataset'], right_on=['method', 'dataset'])

            for column in ('smape', 'duration', 'performance_rank'):
                best = df[['method', column]].groupby('method').mean()[column].idxmin()

                for method in df['method'].unique():
                    t = ttest_ind(
                        df[df['method'] == best][column].values,
                        df[df['method'] == method][column].values
                    )
                    if t.pvalue * len(df['method'].unique()) > 0.05:
                        print(column, method, t)

            df = df.groupby(['dataset', 'method']).mean(numeric_only=True).reset_index()
            aggregated = df.groupby('method').mean(numeric_only=True).join(
                df.groupby('method').std(numeric_only=True),
                lsuffix='_mean',
                rsuffix='_std'
            )
            print(aggregated.sort_index(key=lambda col: col.str.lower()), end='\n\n\n')

            wide = df[['dataset', 'method', 'smape']].pivot(index='dataset', columns='method', values='smape')
            diagram = Diagram(
                wide.to_numpy(),
                treatment_names=wide.columns,
                maximize_outcome=False
            )
            filename = f'cd_{"benchmark" if results == benchmark else "ablation"}-{time}{"_timeout" if with_timeout else ""}{"_nan" if with_missing_values else ""}.pdf'
            diagram.to_file(filename, alpha=.05, adjustment='holm', reverse_x=False)
            for suffix in ('.aux', '.log', '.tex'):
                os.remove(filename.replace('.pdf', suffix))

            raw = df.pivot_table(index='dataset', columns='method', values=['smape', 'duration', 'performance_rank'],
                                 aggfunc='first')

            methods = df['method'].unique()
            print(methods)
            for index, row in raw.iterrows():
                print(index.replace('_', '\\_'), end='')
                for method in methods:
                    print(
                        ' & '
                        f'{row[("smape", method)]:.2f}', ' & ',
                        # f'{row[("performance_rank", method)]:.2f}', ' & ',
                        f'{row[("duration", method)]:.2f}',
                        end=' '
                    )
                print('\\\\')
            pass


analyse_results(with_timeout=True, with_missing_values=True)
analyse_results(with_timeout=True, with_missing_values=False)
analyse_results(with_timeout=False, with_missing_values=True)
