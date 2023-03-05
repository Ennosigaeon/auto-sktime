import os
import re

import numpy as np
import pandas as pd

from autosktime.data.benchmark import *
from autosktime.data.benchmark.base import Benchmark
from autosktime.data.splitter import multiindex_split


def export_benchmark(benchmark: Benchmark, output_folder: str):
    np.random.seed(42)

    X, y = benchmark.get_data()
    train_folds, val_folds, test_folds = benchmark.get_train_test_splits()
    for fold, ((_, train), (_, val), (_, test)) in enumerate(
            zip(train_folds.iterrows(), val_folds.iterrows(), test_folds.iterrows())):
        y_train, y_test, X_train, X_test = multiindex_split(y, X, train=pd.concat((train, val)), test=test)

        name = f"{benchmark.name()}_{fold}".replace("_", "")

        def convert_df(df: pd.DataFrame, y_series: pd.Series) -> str:
            arff_content = [f"@relation {name}"]
            for col in df.columns:
                sanitized_col = re.sub("[^a-zA-Z0-9]+", "", col)
                arff_content.append(f"@ATTRIBUTE sensor_{sanitized_col} TIMESERIES")
            arff_content.append(f"@ATTRIBUTE RUL NUMERIC")
            arff_content.append("@DATA")

            for instance in df.index.levels[0]:
                length = df.loc[instance].shape[0]
                cutoff = min(length, int(np.random.uniform(50.0, max(length, 50.0))))

                values = []
                for col in df.columns:
                    feature = df.loc[instance].loc[0:cutoff][col]
                    if feature.shape[0] > 5000:
                        feature = feature[::feature.shape[0] // 5000]
                    values.append(f'"{(feature.index.astype(str) + "#" + feature.astype(str)).str.cat(sep=" ")}"')

                rul = y_series.loc[instance].loc[cutoff][0]
                if np.isnan(rul):
                    rul = 0
                values.append(str(rul))
                arff_content.append(','.join(values))
            return '\n'.join(arff_content)

        with open(f'{output_folder}/{name}_train.arff', 'w') as f:
            f.write(convert_df(X_train, y_train))
        with open(f'{output_folder}/{name}_test.arff', 'w') as f:
            f.write(convert_df(X_test, y_test))


def score_results(benchmark: Benchmark, input_folder: str):
    input_folder = input_folder + benchmark.name()
    X, y = benchmark.get_data()
    train_folds, val_folds, test_folds = benchmark.get_train_test_splits()

    for fold, file in enumerate(sorted(os.listdir(input_folder))):
        y_pred = pd.read_csv(f'{input_folder}/{file}', header=None)
        _, y_test = multiindex_split(y, train=pd.concat((train_folds.loc[fold], val_folds.loc[fold])),
                                     test=test_folds.loc[fold])

        step_size = abs(y_test.diff().min().min()) * 10

        reconstructed = []
        for (_, count), (_, value) in zip(y_test.groupby(level=0).count().sort_index(ascending=False).iterrows(),
                                          y_pred.iterrows()):
            count = count.iloc[0]
            value = value.iloc[0] - 0
            values = np.clip(
                np.flip(np.arange(0, count) * step_size + value),
                0, benchmark.get_piecewise_cutoff()
            )
            reconstructed.append(values)

        reconstructed = np.hstack(reconstructed)
        benchmark.score_solutions(reconstructed, y_test)

        print(file, benchmark.performance['rmse'])


if __name__ == '__main__':
    benchmark = FemtoBenchmark()

    export_benchmark(benchmark, 'data/autorul')
    score_results(benchmark, 'results/AutoCoevoRUL/')
