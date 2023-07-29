import os
import shutil

import numpy as np
import pandas as pd


def evaluate_autogluon(y: pd.Series, fh: int, max_duration: int, name: str, seed: int, hpo: bool = False):
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    if os.path.exists('./AutogluonModels'):
        shutil.rmtree('./AutogluonModels')

    y = y.reset_index()
    y['item_id'] = 0

    train_data = TimeSeriesDataFrame.from_data_frame(
        y,
        id_column="item_id",
        timestamp_column="index"
    )

    predictor = TimeSeriesPredictor(
        prediction_length=fh,
        quantile_levels=[0.25, 0.75],
        target="y",
        eval_metric="sMAPE",
    )

    # Disable CUDA due to SIGSEV
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    predictor.fit(
        train_data,
        presets="best_quality" if hpo else None,
        time_limit=max_duration,
        random_seed=seed,
        hyperparameter_tune_kwargs={'num_trials': 10000, 'scheduler': 'local', 'searcher': 'auto'} if hpo else None
    )

    predictions = predictor.predict(train_data)
    predictions = predictions.loc[0]
    predictions.index = np.arange(y.index[-1], y.index[-1] + fh) + 1

    y_pred_ints = pd.DataFrame(
        predictions[['0.25', '0.75']].values,
        columns=pd.MultiIndex.from_tuples([('Coverage', 0.5, 'lower'), ('Coverage', 0.5, 'upper')]),
        index=predictions.index
    )
    return predictions['mean'], y_pred_ints
