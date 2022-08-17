import logging
import os
from pathlib import Path

import pandas as pd
import torch
from matplotlib import pyplot as plt

from autosktime.data.benchmark.rul import load_rul, load_rul_splits
from autosktime.data.splitter import multiindex_split
from autosktime.metrics import RootMeanSquaredError
from autosktime.pipeline.components.index import AddIndexComponent, AddSequenceComponent
from autosktime.pipeline.components.nn.dataloader import DataLoaderComponent
from autosktime.pipeline.components.nn.network.rnn import RecurrentNetwork
from autosktime.pipeline.components.nn.trainer import TrainerComponent

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.INFO)

data_dir = os.path.join(Path(__file__).parent.resolve(), 'data', 'rul')
X, y = load_rul(data_dir)
train_folds, val_folds, test_folds = load_rul_splits(data_dir, 10)
target = 'RUL'
features = ['Differenzdruck', 'Vorschub', '__index__', '__sequence__']

for (_, train), (_, val), (_, test) in zip(train_folds.iterrows(), val_folds.iterrows(), test_folds.iterrows()):
    y_train, y_test, X_train, X_test = multiindex_split(y, X, train=train, test=val)

    df_train = pd.concat((X_train.copy(), y_train.copy()), axis=1)
    df_train = AddIndexComponent().transform(df_train)
    df_train = AddSequenceComponent().transform(df_train)
    df_test = pd.concat((X_test.copy(), y_test.copy()), axis=1)
    df_test = AddIndexComponent().transform(df_test)
    df_test = AddSequenceComponent().transform(df_test)

    # Normalize Data using train_df
    target_mean = df_train[target].mean()
    target_stdev = df_train[target].std()
    for c in df_train.columns:
        mean = df_train[c].mean()
        stdev = df_train[c].std()

        df_train[c] = (df_train[c] - mean) / stdev
        df_test[c] = (df_test[c] - mean) / stdev

    # Create train and test dataset
    torch.manual_seed(101)

    batch_size = 512
    window_length = 50
    train_data = DataLoaderComponent(batch_size=batch_size, window_length=window_length, validation_size=0) \
        .fit({'X': df_train[features].values, 'y': df_train[target].values})
    test_data = DataLoaderComponent(batch_size=batch_size, window_length=window_length, validation_size=0) \
        .fit({'X': df_test[features].values, 'y': df_test[target].values})

    X, y = next(iter(train_data.train_loader_))

    print("Features shape:", X.shape)
    print("Target shape:", y.shape)

    # NN training
    model = RecurrentNetwork(
        hidden_size=20,
        num_layers=2,
        dropout=0.3
    ).fit({'X': df_train[features].values})
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    trainer = TrainerComponent(desired_iterations=32, patience=5)

    trainer.fit({
        'train_data_loader': train_data.train_loader_,
        'val_data_loader': test_data.val_loader_,
        'network': model,
        'optimizer': optimizer,
        'device': 'cuda'
    })

    y_pred = trainer.predict({'test_data_loader': test_data.val_loader_}) * target_stdev + target_mean
    out = pd.concat((X_test.copy(), y_test.copy()), axis=1)
    out['Model Forecast'] = y_pred
    out.to_csv('forecast_test.csv')
    out[['RUL', 'Model Forecast']].plot()
    plt.show()
    print(f'Test Score: {RootMeanSquaredError(start=50)(out[["Model Forecast"]], out[["RUL"]])}')

    y_pred = trainer.predict({'test_data_loader': train_data.val_loader_}) * target_stdev + target_mean
    out = pd.concat((X_train.copy(), y_train.copy()), axis=1)
    out['Model Forecast'] = y_pred
    out.to_csv('forecast_train.csv')
    out[['RUL', 'Model Forecast']].plot()
    plt.show()
    print(f'Train Score: {RootMeanSquaredError(start=50)(out[["Model Forecast"]], out[["RUL"]])}')

    break
