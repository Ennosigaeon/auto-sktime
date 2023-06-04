import warnings
from typing import Type

import pandas as pd
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
from sktime.forecasting.fbprophet import Prophet

from scripts.benchmark.util import generate_fh

warnings.filterwarnings("ignore")


def evaluate(y: pd.Series, clazz: Type[BaseForecaster], fh_: int):
    fh = ForecastingHorizon(generate_fh(y.index, fh_), is_relative=False)

    orig_freq = pd.infer_freq(y.index)
    # See https://github.com/pandas-dev/pandas/issues/38914
    freq = 'M' if orig_freq == 'MS' else orig_freq
    y.index = y.index.to_period(freq)

    forecaster = clazz()
    forecaster.fit(y)

    y_pred = forecaster.predict(fh)
    y_pred_ints = forecaster.predict_interval(coverage=0.5)

    return y_pred, y_pred_ints


def evaluate_arima(y: pd.Series, fh: int):
    return evaluate(y, AutoARIMA, fh_=fh)


def evaluate_prophet(y: pd.Series, fh: int):
    return evaluate(y, Prophet, fh_=fh)
