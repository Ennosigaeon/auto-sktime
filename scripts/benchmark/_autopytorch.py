from typing import Optional

import numpy as np
import pandas as pd
from pandas._libs import OutOfBoundsDatetime
from sklearn.preprocessing import LabelEncoder


def evaluate_autopytorch(
        y: pd.Series,
        X_train: Optional[pd.DataFrame],
        X_test: Optional[pd.DataFrame],
        fh: int,
        max_duration: int,
        name: str,
        seed: int,
):
    from autoPyTorch.api.time_series_forecasting import TimeSeriesForecastingTask
    from autoPyTorch.datasets.time_series_dataset import TimeSeriesSequence

    if isinstance(y.index, pd.MultiIndex):
        series_idx = LabelEncoder().fit_transform(y.index.get_level_values(0))
        if X_train is None:
            X_train = pd.DataFrame({'series': series_idx})
        else:
            X_train['series'] = series_idx

    try:
        api = TimeSeriesForecastingTask()
        api.search(
            y_train=[y],
            X_train=[X_train] if X_train is not None else None,
            known_future_features=X_test.columns if X_test is not None else None,
            start_times=[y.index[0][1] if isinstance(y.index, pd.MultiIndex) else y.index[0]],
            series_idx=['series'] if isinstance(y.index, pd.MultiIndex) else None,
            optimize_metric='mean_MASE_forecasting',
            n_prediction_steps=fh,
            dataset_name=name,
            total_walltime_limit=max_duration,
            seed=seed,

            freq='1H',  # freq has to be provided as otherwise the code crashes for longer time series
        )
    except (ValueError, OutOfBoundsDatetime):
        api = TimeSeriesForecastingTask()
        api.search(
            y_train=[y],
            X_train=[X_train] if X_train is not None else None,
            known_future_features=X_test.columns if X_test is not None else None,
            start_times=[y.index[0][1] if isinstance(y.index, pd.MultiIndex) else y.index[0]],
            series_idx=['series'] if isinstance(y.index, pd.MultiIndex) else None,
            freq='1H',  # freq has to be provided as otherwise the code crashes for longer time series
            optimize_metric='mean_MASE_forecasting',
            n_prediction_steps=fh,
            dataset_name=name,
            total_walltime_limit=max_duration,
            seed=seed
        )

    if X_test is not None:
        test_data = [TimeSeriesSequence(X_test.values, y.values[:X_test.shape[0]], X_test=X_test.values)]
    else:
        test_data = api.dataset.generate_test_seqs()
    y_pred = api.predict(test_data)

    if y.shape[1] == 1:
        y_pred = pd.DataFrame(y_pred.T, columns=y.columns)
    else:
        y_pred = pd.DataFrame(y_pred[0], columns=y.columns)
    y_pred_ints = pd.DataFrame(
        np.tile(y_pred.values, 2),
        columns=pd.MultiIndex.from_tuples(
            [(f'Coverage_{col}', 0.5, 'lower') for col in y_pred.columns] +
            [(f'Coverage_{col}', 0.5, 'upper') for col in y_pred.columns]
        ),
        index=y_pred.index
    )

    return y_pred, y_pred_ints
