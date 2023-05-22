import warnings
from typing import Type

import numpy as np
import pandas as pd
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.fbprophet import Prophet

warnings.filterwarnings("ignore")


def evaluate(y: pd.Series, clazz: Type[BaseForecaster], fh: int):
    fh = np.arange(1, fh + 1)

    forecaster = clazz()
    forecaster.fit(y)

    y_pred = forecaster.predict(fh)
    y_pred_ints = forecaster.predict_interval(coverage=0.5)

    return y_pred, y_pred_ints


def evaluate_arima(y: pd.Series, fh: int):
    return evaluate(y, AutoARIMA, fh=fh)


def evaluate_prophet(y: pd.Series, fh: int):
    return evaluate(y, Prophet, fh=fh)
