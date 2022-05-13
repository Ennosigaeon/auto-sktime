import warnings
from typing import Optional, Dict

import numpy as np

from autosktime.constants import FORECAST_TASK, UNIVARIATE_FORECAST, UNIVARIATE_EXOGENOUS_FORECAST
from sktime.forecasting.model_selection._split import ACCEPTED_Y_TYPES
from sktime.performance_metrics.forecasting import (
    MeanAbsolutePercentageError as MeanAbsolutePercentageError_,
    MedianAbsolutePercentageError as MedianAbsolutePercentageError_
)
# noinspection PyProtectedMember
from sktime.performance_metrics.forecasting._classes import _BaseForecastingErrorMetric

BaseMetric = _BaseForecastingErrorMetric


class _BoundedMetricMixin:
    optimum: float
    worst_possible_result: float


def calculate_loss(
        solution: ACCEPTED_Y_TYPES,
        prediction: ACCEPTED_Y_TYPES,
        task_type: int,
        metric: BaseMetric,
) -> float:
    """
    Returns a loss (a magnitude that allows casting the
    optimization problem as a minimization one) for the
    given BaseMetric object

    Parameters
    ----------
    solution: Union[pd.Series, pd.DataFrame, np.ndarray, pd.Index]
        The ground truth of the targets
    prediction: Union[pd.Series, pd.DataFrame, np.ndarray, pd.Index]
        The best estimate from the model, of the given targets
    task_type: int
        To understand the problem task
    metric: BaseMetric
        Object that host a function to calculate how good the
        prediction is according to the solution.

    Returns
    -------
    float
        The loss value
    """

    if task_type in FORECAST_TASK:
        score = metric(solution, prediction)
    else:
        raise NotImplementedError('Scoring of non FORECAST_TASK not supported')

    if metric.greater_is_better:
        if hasattr(metric, 'optimum'):
            return metric.optimum - score
        else:
            raise ValueError(f'Metric {type(metric)} has to implement {_BoundedMetricMixin}')
    else:
        return score


def get_cost_of_crash(metric: BaseMetric) -> float:
    if metric.greater_is_better and hasattr(metric, 'optimum') and hasattr(metric, 'worst_possible_result'):
        return metric.optimum - metric.worst_possible_result
    elif not metric.greater_is_better and hasattr(metric, 'worst_possible_result'):
        return metric.worst_possible_result
    else:
        raise ValueError(f'Metric {type(metric)} has to implement {_BoundedMetricMixin}')


class MeanAbsolutePercentageError(_BoundedMetricMixin, MeanAbsolutePercentageError_):

    def __init__(self):
        super().__init__()
        self.optimum = 0.
        self.worst_possible_result = 1.

    def __call__(
            self,
            y_true: ACCEPTED_Y_TYPES,
            y_pred: ACCEPTED_Y_TYPES,
            horizon_weight: Optional[np.ndarray] = None
    ) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            return super().__call__(y_true, y_pred, horizon_weight=horizon_weight)


class MedianAbsolutePercentageError(_BoundedMetricMixin, MedianAbsolutePercentageError_):

    def __init__(self):
        super().__init__()
        self.optimum = 0.
        self.worst_possible_result = 1.

    def __call__(
            self,
            y_true: ACCEPTED_Y_TYPES,
            y_pred: ACCEPTED_Y_TYPES,
            horizon_weight: Optional[np.ndarray] = None
    ) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            return super().__call__(y_true, y_pred, horizon_weight=horizon_weight)


default_metric_for_task: Dict[int, BaseMetric] = {
    UNIVARIATE_FORECAST: MeanAbsolutePercentageError(),
    UNIVARIATE_EXOGENOUS_FORECAST: MeanAbsolutePercentageError(),
}
