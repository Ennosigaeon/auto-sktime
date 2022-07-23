from typing import Optional, List

import numpy as np
import pandas as pd
from sktime.forecasting.compose import EnsembleForecaster
from sktime.forecasting.compose._ensemble import _check_aggfunc

from autosktime.constants import SUPPORTED_Y_TYPES
from autosktime.pipeline.components.base import AutoSktimePredictor
from autosktime.pipeline.util import NotVectorizedMixin


# noinspection PyAbstractClass
class PrefittedEnsembleForecaster(NotVectorizedMixin, EnsembleForecaster, AutoSktimePredictor):

    def _fit(self, y, X=None, fh=None):
        self.forecasters_ = self.forecasters

    def _predict(self, fh, X=None):
        y_pred = pd.concat(self._predict_forecasters(fh, X), axis=1)
        y_pred = _aggregate(y=y_pred, aggfunc=self.aggfunc, weights=self.weights)

        return y_pred


def _aggregate(y: SUPPORTED_Y_TYPES, aggfunc: str, weights: Optional[List[float]]) -> SUPPORTED_Y_TYPES:
    if weights is None:
        aggfunc = _check_aggfunc(aggfunc, weighted=False)
        y_agg = aggfunc(y, axis=1)
    else:
        aggfunc = _check_aggfunc(aggfunc, weighted=True)
        y_agg = aggfunc(y, axis=1, weights=np.array(weights))

    if isinstance(y, pd.Series):
        return pd.Series(y_agg, index=y.index, name=y.name)
    elif isinstance(y, pd.DataFrame):
        return pd.DataFrame(y_agg, index=y.index, columns=y.columns[[0]])
