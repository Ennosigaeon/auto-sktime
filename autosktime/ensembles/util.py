import logging
import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import EnsembleForecaster
from sktime.forecasting.compose._ensemble import _check_aggfunc
from typing import Optional, List, Tuple

from autosktime.constants import SUPPORTED_Y_TYPES
from autosktime.pipeline.components.base import AutoSktimePredictor
from autosktime.pipeline.util import NotVectorizedMixin


# noinspection PyAbstractClass
class PrefittedEnsembleForecaster(NotVectorizedMixin, EnsembleForecaster, AutoSktimePredictor):
    _tags = {
        'capability:pred_int': True
    }

    def __init__(
            self,
            forecasters: List[Tuple[str, AutoSktimePredictor]],
            n_jobs: int = None,
            aggfunc: str = "mean",
            weights: List[float] = None
    ):
        super().__init__(forecasters, n_jobs=n_jobs, aggfunc=aggfunc, weights=weights)
        self.logger = logging.getLogger(__name__)

    def _fit(self, y, X=None, fh=None):
        self.forecasters_ = self.forecasters

    def _predict(self, fh: ForecastingHorizon, X: pd.DataFrame = None):
        y_pred, valid = self._predict_forecasters(fh, X)
        weights = [self.weights[i] for i in valid] if self.weights is not None else None
        y_pred = _aggregate(y=y_pred, aggfunc=self.aggfunc, weights=weights)

        return y_pred

    def _predict_forecasters(
            self,
            fh: ForecastingHorizon = None,
            X: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, List[int]]:
        res = []
        valid = []

        for i, (_, forecaster) in enumerate(self.forecasters_):
            try:
                res.append(forecaster.predict(fh=fh, X=X))
                valid.append(i)
            except KeyboardInterrupt:
                raise
            except Exception:
                self.logger.warning(f'Failed to create predictions with forecaster {forecaster}:', exc_info=True)

        return pd.concat(res, axis=1), valid

    def _predict_interval(self, fh: ForecastingHorizon = None, X: pd.DataFrame = None, coverage: float = 0.90):
        y_pred_int, valid = self._predict_interval_forecasters(fh, X, coverage)
        weights = [self.weights[i] for i in valid] if self.weights is not None else None
        y_pred_int = _aggregate(y=y_pred_int, aggfunc=self.aggfunc, weights=weights)

        return y_pred_int

    def _predict_interval_forecasters(
            self,
            fh: ForecastingHorizon = None,
            X: pd.DataFrame = None,
            coverage: float = 0.90
    ) -> Tuple[pd.DataFrame, List[int]]:
        res = []
        valid = []

        for i, (_, forecaster) in enumerate(self.forecasters_):
            try:
                res.append(forecaster.predict_interval(fh=fh, X=X, coverage=coverage))
                valid.append(i)
            except KeyboardInterrupt:
                raise
            except Exception:
                self.logger.warning(f'Failed to create predictions with forecaster {forecaster}:', exc_info=True)

        return pd.concat(res, axis=1), valid


def _aggregate(y: SUPPORTED_Y_TYPES, aggfunc: str, weights: Optional[List[float]]) -> SUPPORTED_Y_TYPES:
    columns = y.columns.unique()
    res = np.empty((y.shape[0], len(columns)))

    for i, col in enumerate(columns):
        if weights is None:
            aggfunc_ = _check_aggfunc(aggfunc, weighted=False)
            res[:, i] = aggfunc_(y[col], axis=1)
        else:
            aggfunc_ = _check_aggfunc(aggfunc, weighted=True)
            res[:, i] = aggfunc_(y[col], axis=1, weights=np.array(weights))

    y_agg = pd.DataFrame(res, columns=columns, index=y.index)

    if isinstance(y, pd.Series):
        return y_agg[y.name]
    elif isinstance(y, pd.DataFrame):
        return y_agg
