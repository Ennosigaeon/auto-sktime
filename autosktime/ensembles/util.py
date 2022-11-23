import logging
import numpy as np
import pandas as pd
from sktime.forecasting.compose import EnsembleForecaster
from sktime.forecasting.compose._ensemble import _check_aggfunc
from typing import Optional, List, Tuple

from autosktime.constants import SUPPORTED_Y_TYPES
from autosktime.pipeline.components.base import AutoSktimePredictor
from autosktime.pipeline.util import NotVectorizedMixin


# noinspection PyAbstractClass
class PrefittedEnsembleForecaster(NotVectorizedMixin, EnsembleForecaster, AutoSktimePredictor):

    def __init__(
            self,
            forecasters: List[AutoSktimePredictor],
            n_jobs: int = None,
            aggfunc: str = "mean",
            weights: List[float] = None
    ):
        super().__init__(forecasters, n_jobs=n_jobs, aggfunc=aggfunc, weights=weights)
        self.logger = logging.getLogger(__name__)

    def _fit(self, y, X=None, fh=None):
        self.forecasters_ = self.forecasters

    def _predict(self, fh, X=None):
        y_pred, valid = self._predict_forecasters(fh, X)
        weights = [self.weights[i] for i in valid] if self.weights is not None else None
        y_pred = _aggregate(y=y_pred, aggfunc=self.aggfunc, weights=weights)

        return y_pred

    def _predict_forecasters(self, fh=None, X=None) -> Tuple[pd.Series, List[int]]:
        res = []
        valid = []

        for i, forecaster in enumerate(self.forecasters_):
            try:
                res.append(forecaster.predict(fh=fh, X=X))
                valid.append(i)
            except KeyboardInterrupt:
                raise
            except Exception:
                self.logger.warning(f'Failed to create predictions with forecaster {forecaster}:', exc_info=True)

        return pd.concat(res, axis=1), valid


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
