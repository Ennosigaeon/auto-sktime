from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss, MultiLoss
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


def evaluate_temporal_fusion_transformer(
        y: pd.Series,
        X_train: Optional[pd.DataFrame],
        X_test: Optional[pd.DataFrame],
        fh: int,
        max_duration: int,
        name: str,
        seed: int
):
    if X_train is not None:
        X_train.columns = [f'col_{i}' for i in range(X_train.shape[1])]
        X_test.columns = [f'col_{i}' for i in range(X_train.shape[1])]

    y_test = pd.DataFrame(data={col: 0 for col in y.columns}, index=X_test.index if X_test is not None else range(fh))

    train = (y.join(X_train) if X_train is not None else y).reset_index()
    test = (y_test.join(X_test) if X_test is not None else y_test).reset_index()

    data = pd.concat((train, test), ignore_index=True)
    data['index'] = data.index
    if 'series' not in data.columns:
        data['series'] = 0

    training = TimeSeriesDataSet(
        data[lambda x: x['index'] < train.shape[0]],
        time_idx='index',
        target=y.columns.tolist(),
        group_ids=['series'],
        max_prediction_length=fh,
        time_varying_known_categoricals=(
            X_train.select_dtypes(exclude='number').columns.tolist() if X_train is not None else []
        ),
        time_varying_known_reals=['index'] + (
            X_train.select_dtypes(include='number').columns.tolist() if X_train is not None else []
        ),
        time_varying_unknown_reals=y.columns.tolist()
    )

    testing = TimeSeriesDataSet.from_dataset(
        training, data,
        predict=False,
        min_prediction_idx=training.index.time.max() + 1,
        stop_randomization=True
    )

    train_dataloader = training.to_dataloader(train=True)
    test_dataloader = testing.to_dataloader(train=False)

    logger = TensorBoardLogger(f"temporal_fusion/{name}/{seed}/")
    early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=1e-7, patience=10, verbose=False, mode="min")

    pl.seed_everything(seed)
    trainer = pl.Trainer(
        max_time=timedelta(seconds=max_duration),
        gpus=0,
        callbacks=[early_stop_callback],
        logger=logger,
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        loss=MultiLoss([QuantileLoss([0.5])] * len(y.columns))
    )
    print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")

    # fit the model on the data - redefine the model with the correct learning rate if necessary
    trainer.fit(tft, train_dataloaders=train_dataloader)

    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    predictions = best_tft.predict(test_dataloader)
    predictions = np.array([y_hat.numpy().mean(axis=1)[-12:] for y_hat in predictions]).T

    return pd.DataFrame(predictions, columns=y.columns), pd.DataFrame(predictions, columns=y.columns)
