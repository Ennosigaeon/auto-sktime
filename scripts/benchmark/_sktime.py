import warnings
from typing import Type, Optional

import numpy as np
import pandas as pd
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.naive import NaiveForecaster

from scripts.benchmark.util import generate_fh, fix_frequency

warnings.filterwarnings("ignore")


def _evaluate(
        y: pd.Series,
        X_train: Optional[pd.DataFrame],
        X_test: Optional[pd.DataFrame],
        clazz: Type[BaseForecaster],
        fh_: int,
        max_duration: int
):
    fix_frequency(y, X_train, X_test)
    if isinstance(y.index, pd.MultiIndex):
        fh = ForecastingHorizon(np.arange(1, fh_ + 1), is_relative=True)
    else:
        fh = ForecastingHorizon(generate_fh(y.index, fh_), is_relative=False)

    forecaster = clazz()
    forecaster.fit(y, X_train)

    y_pred = forecaster.predict(fh, X_test)
    y_pred_ints = forecaster.predict_interval(X=X_test, coverage=0.5)

    return y_pred, y_pred_ints


def evaluate_arima(
        y: pd.Series,
        X_train: Optional[pd.DataFrame],
        X_test: Optional[pd.DataFrame],
        fh: int,
        max_duration: int,
        name: str,
        seed: int,
):
    return _evaluate(y, X_train, X_test, AutoARIMA, fh_=fh, max_duration=max_duration)


def evaluate_prophet(
        y: pd.Series,
        X_train: Optional[pd.DataFrame],
        X_test: Optional[pd.DataFrame],
        fh: int,
        max_duration: int,
        name: str,
        seed: int,
):
    return _evaluate(y, X_train, X_test, Prophet, fh_=fh, max_duration=max_duration)


def evaluate_naive(
        y: pd.Series,
        X_train: Optional[pd.DataFrame],
        X_test: Optional[pd.DataFrame],
        fh: int,
        max_duration: int,
        name: str,
        seed: int,
):
    return _evaluate(y, X_train, X_test, NaiveForecaster, fh_=fh, max_duration=max_duration)


def evaluate_ets(
        y: pd.Series,
        X_train: Optional[pd.DataFrame],
        X_test: Optional[pd.DataFrame],
        fh: int,
        max_duration: int,
        name: str,
        seed: int,
):
    return _evaluate(y, X_train, X_test, AutoETS, fh_=fh, max_duration=max_duration)