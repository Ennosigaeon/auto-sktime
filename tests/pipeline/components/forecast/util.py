from typing import Type, Tuple

import numpy as np
import pandas as pd

import sktime.datasets
from autosktime.pipeline.components.base import AutoSktimePredictor
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split


def get_dataset(dataset: str = 'airline', add_NaNs: bool = False, train_size_maximum: int = 150):
    ds = getattr(sktime.datasets, f'load_{dataset}')()

    if isinstance(ds, tuple):
        X, y = ds
    else:
        X, y = None, ds

    train_size = min(int(y.shape[0] / 3. * 2.), train_size_maximum)

    if X is None:
        y_train, y_test = temporal_train_test_split(y, X=X, train_size=train_size)
        X_train, X_test = None, None
    else:
        y_train, y_test, X_train, X_test = temporal_train_test_split(y, X=X, train_size=train_size)

    if add_NaNs:
        # TODO missing
        pass

    fh = ForecastingHorizon(np.arange(1, y.shape[0] - train_size + 1))

    return X_train, y_train, X_test, y_test, fh


def _test_forecaster(
        forecaster: Type[AutoSktimePredictor],
        dataset: str = 'airline',
        train_size_maximum: int = 150
) -> Tuple[pd.Series, pd.Series]:
    X_train, y_train, X_test, y_test, fh = get_dataset(dataset=dataset, train_size_maximum=train_size_maximum)

    configuration_space = forecaster.get_hyperparameter_search_space()
    default_config = configuration_space.get_default_configuration()

    # noinspection PyArgumentList
    forecaster = forecaster(random_state=0, **default_config)

    predictor = forecaster.fit(y_train, X=X_train)

    predictions = predictor.predict(fh=fh, X=X_test)
    return predictions, y_test
