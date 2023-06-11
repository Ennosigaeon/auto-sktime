import warnings
from typing import Type

import pandas as pd
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
from sktime.forecasting.fbprophet import Prophet

from scripts.benchmark.util import generate_fh, fix_frequency

warnings.filterwarnings("ignore")


def evaluate(y: pd.Series, clazz: Type[BaseForecaster], fh_: int, max_duration: int):
    fix_frequency(y)
    fh = ForecastingHorizon(generate_fh(y.index, fh_), is_relative=False)

    forecaster = clazz()
    forecaster.fit(y)

    y_pred = forecaster.predict(fh)
    y_pred_ints = forecaster.predict_interval(coverage=0.5)

    return y_pred, y_pred_ints


def evaluate_arima(y: pd.Series, fh: int, max_duration: int):
    return evaluate(y, AutoARIMA, fh_=fh, max_duration=max_duration)


def evaluate_prophet(y: pd.Series, fh: int, max_duration: int):
    return evaluate(y, Prophet, fh_=fh, max_duration=max_duration)
