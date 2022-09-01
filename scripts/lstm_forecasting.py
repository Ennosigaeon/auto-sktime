import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from autosktime.data.benchmark.rul import load_rul, load_rul_splits
from autosktime.data.splitter import multiindex_split
from autosktime.metrics import RootMeanSquaredError
from autosktime.pipeline.components.index import AddIndexComponent
from autosktime.pipeline.components.nn.dataloader import DataLoaderComponent
from autosktime.pipeline.components.nn.network.cnn import CNN
from autosktime.pipeline.components.nn.trainer import TrainerComponent
from autosktime.util.backend import ConfigContext

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.INFO)


def find_runs(x):
    # find run starts
    loc_run_start = np.empty(x.shape[0], dtype=bool)
    loc_run_start[0] = True
    np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
    run_starts = np.nonzero(loc_run_start)[0]

    # find run lengths
    run_lengths = np.diff(np.append(run_starts, x.shape[0]))

    return run_lengths


def prepare_data(X_train, y_train, X_val, y_val, X_test, y_test):
    df_train = pd.concat((X_train.copy(), y_train.copy()), axis=1)
    df_train = AddIndexComponent().transform(df_train)

    df_val = pd.concat((X_val.copy(), y_val.copy()), axis=1)
    df_val = AddIndexComponent().transform(df_val)

    df_test = pd.concat((X_test.copy(), y_test.copy()), axis=1)
    df_test = AddIndexComponent().transform(df_test)

    # Normalize Data using train_df
    for c in df_train.columns:
        mean = df_train[c].mean()
        stdev = df_train[c].std()

        df_train[c] = (df_train[c] - mean) / stdev
        df_val[c] = (df_val[c] - mean) / stdev
        df_test[c] = (df_test[c] - mean) / stdev

    return df_train, df_val, df_test


def create_data_loaders(df_train, df_val, df_test):
    context: ConfigContext = ConfigContext.instance()
    context.set_config(None, {
        'panel_sizes': find_runs(df_train.index.get_level_values(0)),
        'panel_sizes_val': find_runs(df_val.index.get_level_values(0))
    })

    batch_size = 512
    window_length = 50
    train_data = DataLoaderComponent(batch_size=batch_size, window_length=window_length, validation_size=0) \
        .fit({
        'X': df_train[features].values, 'y': df_train[target].values,
        'X_val': df_val[features].values, 'y_val': df_val[target].values
    })

    context.set_config(None, {
        'panel_sizes': find_runs(df_test.index.get_level_values(0)),
    })
    test_data = DataLoaderComponent(batch_size=batch_size, window_length=window_length, validation_size=0) \
        .fit({'X': df_test[features].values, 'y': df_test[target].values})

    return train_data, test_data


data_dir = os.path.join(Path(__file__).parent.resolve(), 'data', 'rul')
X, y = load_rul(data_dir)
train_folds, val_folds, test_folds = load_rul_splits(data_dir, 10)
target = 'RUL'
features = ['Differenzdruck', 'Vorschub', '__index__']

for (_, train), (_, val), (_, test) in zip(train_folds.iterrows(), val_folds.iterrows(), test_folds.iterrows()):
    y_train, y_val, X_train, X_val = multiindex_split(y, X, train=train, test=val)
    _, y_test, _, X_test = multiindex_split(y, X, train=train, test=test)
    df_train, df_val, df_test = prepare_data(X_train, y_train, X_val, y_val, X_test, y_test)

    # Create train and test dataset
    torch.manual_seed(101)

    train_data, test_data = create_data_loaders(df_train, df_val, df_test)

    X, y = next(iter(train_data.train_loader_))

    print("Features shape:", X.shape)
    print("Target shape:", y.shape)

    # NN training
    model = CNN(
        num_layers=5,
        kernel_size=20,
        num_filters=16,
        pool_size=1,
        dropout=0.3
    ).fit({'X': df_train[features].values})
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = TrainerComponent(iterations=4, patience=5)

    trainer.fit({
        'train_data_loader': train_data.train_loader_,
        'val_data_loader': train_data.val_loader_,
        'network': model,
        'optimizer': optimizer,
        'device': 'cuda'
    })

    target_mean = y_train['RUL'].mean()
    target_stdev = y_train['RUL'].std()

    y_pred = trainer.predict({'test_data_loader': test_data.val_loader_}) * target_stdev + target_mean
    out = pd.concat((X_test.copy(), y_test.copy()), axis=1)
    out['Model Forecast'] = y_pred
    out.to_csv('forecast_test.csv')
    out[['RUL', 'Model Forecast']].plot()
    plt.show()
    print(f'Test Score: {RootMeanSquaredError(start=50)(out[["Model Forecast"]], out[["RUL"]])}')

    y_pred = trainer.predict({'test_data_loader': train_data.val_loader_}) * target_stdev + target_mean
    out = pd.concat((X_val.copy(), y_val.copy()), axis=1)
    out['Model Forecast'] = y_pred
    out.to_csv('forecast_train.csv')
    out[['RUL', 'Model Forecast']].plot()
    plt.show()
    print(f'Train Score: {RootMeanSquaredError(start=50)(out[["Model Forecast"]], out[["RUL"]])}')

    break
