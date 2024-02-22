import os

import numpy as np
import pandas as pd
from critdd import Diagram
from scipy.stats import ttest_ind

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 10000)
pd.set_option("display.precision", 2)
pd.set_option('use_inf_as_na', True)


def analyse_results(
        with_timeout: bool = True,
        with_missing_values: bool = True,
        imputation_method: str = 'worst',
        metric: str = 'mase',
        max_duration: int = 360
):
    print(f'with_timeout = {with_timeout}, with_missing_values = {with_missing_values}')

    benchmark = {
        'auto-pytorch': pd.read_csv(f'results2/auto-pytorch/_result.csv'),
        'auto-sktime': pd.read_csv(f'results2/auto-sktime/_result.csv'),
        'autogluon': pd.read_csv('results2/autogluon/_result.csv'),
        'autots_random': pd.read_csv(f'results2/autots_random/_result.csv'),
        'hyperts': pd.read_csv(f'results2/hyperts/_result.csv'),
        'pmdarima': pd.read_csv('results2/pmdarima/_result.csv'),
        'pyaf': pd.read_csv('results2/pyaf/_result.csv'),
        'naive': pd.read_csv('results2/naive/_result.csv'),
        'ets': pd.read_csv('results2/ets/_result.csv'),
        'tft': pd.read_csv(f'results2/tft/_result.csv'),
        'deepar': pd.read_csv(f'results2/deepar/_result.csv'),
    }
    ablation_study = {
        'auto-sktime': pd.read_csv(f'results2/auto-sktime/_result.csv'),
        'auto-sktime_templates': pd.read_csv(f'results2/auto-sktime_templates/_result.csv'),
        'auto-sktime_multi_fidelity': pd.read_csv(f'results2/auto-sktime_multi_fidelity/_result.csv'),
        'auto-sktime_warm_starting': pd.read_csv(f'results2/auto-sktime_warm_starting/_result.csv'),
        'auto-pytorch': pd.read_csv(f'results2/auto-pytorch/_result.csv'),
        'naive': pd.read_csv('results2/naive/_result.csv'),
    }

    for results in (ablation_study, benchmark):
        print(f'###############\n'
              f'Benchmark {results == benchmark}, Timeout {with_timeout}, Missing Values {with_missing_values}\n'
              f'###############\n')

        df = pd.concat(results.values())

        if not with_missing_values:
            nan = df[(df['method'] == 'pmdarima') & np.isinf(df[metric])]['dataset'].unique()
            df = df[~df['dataset'].isin(nan)]

        if with_timeout:
            df.loc[(df['duration'] > max_duration), metric] = np.inf

        df = df.drop(columns=['seed']).round(5)
        if imputation_method == 'worst':
            missing = pd.isna(df).any(axis=1)
            for m in ('mase', 'sMAPE', 'RMSE'):
                worst = (1.05 * df[['dataset', m]].dropna().groupby('dataset').max()).loc[df.loc[missing, 'dataset']]
                worst.index = df.loc[missing, m].index
                df.loc[missing, m] = worst
        elif imputation_method == 'naive':
            imputation = df[df['method'] == 'naive']
            for method in df['method'].unique():
                for m in ('mase', 'sMAPE', 'RMSE'):
                    missing = df[df['method'] == method][m] == np.inf
                    df.loc[(df['method'] == method) & (df[m] == np.inf), m] = imputation[missing][m]

        df = df[df['method'] != 'naive']

        print('Number of timeouts')
        print(df.loc[(df['duration'] > max_duration), 'method'].value_counts())

        if with_timeout:
            df['duration'] = df['duration'].clip(upper=max_duration)

        rank = df.groupby(['dataset', 'method']).mean(numeric_only=True).groupby('dataset').rank()[
            metric].reset_index().rename(columns={metric: 'performance_rank'})
        df = pd.merge(df, rank, how='left', left_on=['method', 'dataset'], right_on=['method', 'dataset'])

        for column in (metric, 'duration', 'performance_rank'):
            best = df[['method', column]].groupby('method').mean()[column].idxmin()

            for method in df['method'].unique():
                t = ttest_ind(
                    df[df['method'] == best][column].values,
                    df[df['method'] == method][column].values
                )
                if t.pvalue * len(df['method'].unique()) > 0.05:
                    print(column, method, t)
        raw = df.copy()

        df = df.groupby(['dataset', 'method']).mean(numeric_only=True).reset_index()
        aggregated = df.groupby('method').mean(numeric_only=True).join(
            df.groupby('method').std(numeric_only=True),
            lsuffix='_mean',
            rsuffix='_std'
        )
        print(aggregated.sort_index(key=lambda col: col.str.lower()), end='\n\n\n')

        wide = df[['dataset', 'method', metric]].pivot(index='dataset', columns='method', values=metric)
        diagram = Diagram(
            wide.to_numpy(),
            treatment_names=wide.columns,
            maximize_outcome=False
        )
        filename = f'cd_{"benchmark" if results == benchmark else "ablation"}{"_timeout" if with_timeout else ""}{"_nan" if with_missing_values else ""}.pdf'
        diagram.to_file(filename, alpha=5.0, adjustment='bonferroni', reverse_x=False)
        for suffix in ('.aux', '.log', '.tex'):
            os.remove(filename.replace('.pdf', suffix))

        raw = df.pivot_table(index='dataset', columns='method', values=[metric, 'duration', 'performance_rank'],
                             aggfunc='first')
        raw.round(2).to_csv('raw.csv')

        methods = df['method'].unique()
        print(methods)
        for index, row in raw.iterrows():
            m = index.replace('_', '\\_').replace('\\_utilization', '')[:-4]
            print(f"{m:30s}", end='')
            for method in methods:
                print(
                    ' & ',
                    '--  ' if (results[method][results[method]['dataset'] == index][
                        metric]).isna().all() else f'{row[(metric, method)]:.2f}',
                    ' & ',
                    # f'{row[("performance_rank", method)]:.2f}', ' & ',
                    f'{int(row[("duration", method)])}',
                    end=' '
                )
            print('\\\\')
        pass


analyse_results(with_timeout=True, with_missing_values=True)
analyse_results(with_timeout=True, with_missing_values=False)
analyse_results(with_timeout=False, with_missing_values=True)
