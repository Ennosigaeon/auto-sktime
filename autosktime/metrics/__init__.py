from abc import abstractmethod
from typing import Any, Optional

import numpy as np
import pandas as pd
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

from autosktime.constants import FORECAST_TASK


class Scorer:
    def __init__(
            self,
            name: str,
            optimum: float,
            worst_possible_result: float,
            sign: float,
            kwargs: Any
    ) -> None:
        self.name = name
        self.optimum = optimum
        self._kwargs = kwargs

        self._worst_possible_result = worst_possible_result
        self._sign = sign

    @abstractmethod
    def __call__(
            self,
            y_true: pd.Series,
            y_pred: pd.Series,
            horizon_weight: Optional[np.ndarray] = None
    ) -> float:
        raise NotImplementedError()

    def __repr__(self) -> str:
        return self.name


def calculate_loss(
        solution: pd.Series,
        prediction: pd.Series,
        task_type: int,
        metric: Scorer,
) -> float:
    """
    Returns a loss (a magnitude that allows casting the
    optimization problem as a minimization one) for the
    given Auto-Sklearn Scorer object

    Parameters
    ----------
    solution: np.ndarray
        The ground truth of the targets
    prediction: np.ndarray
        The best estimate from the model, of the given targets
    task_type: int
        To understand if the problem task is classification
        or regression
    metric: Scorer
        Object that host a function to calculate how good the
        prediction is according to the solution.

    Returns
    -------
    float or Dict[str, float]
        A loss function for each of the provided scorer objects
    """

    if task_type in FORECAST_TASK:
        score = metric(solution, prediction)
    else:
        raise NotImplementedError('Scoring of non FORECAST_TASK not supported')

    if metric._sign > 0:
        rval = score
    else:
        rval = metric.optimum - score
    return rval


class MeanAbsolutePercentageError(Scorer):

    def __init__(self, **kwargs):
        super().__init__('mape', 0., 1., 1., kwargs)

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, horizon_weight: Optional[np.ndarray] = None) -> float:
        return mean_absolute_percentage_error(y_true, y_pred, horizon_weight=horizon_weight, **self._kwargs)
