import warnings
from typing import Optional, Dict

import numpy as np
from sktime.performance_metrics.forecasting import (
    MeanAbsolutePercentageError as MeanAbsolutePercentageError_,
    MedianAbsolutePercentageError as MedianAbsolutePercentageError_,
    MeanAbsoluteScaledError as MeanAbsoluteScaledError_
)
# noinspection PyProtectedMember
from sktime.performance_metrics.forecasting._classes import _BaseForecastingErrorMetric, BaseForecastingErrorMetric

from autosktime.constants import FORECAST_TASK, UNIVARIATE_FORECAST, UNIVARIATE_EXOGENOUS_FORECAST, MAXINT, \
    SUPPORTED_Y_TYPES, MULTIVARIATE_FORECAST, MULTIVARIATE_EXOGENOUS_FORECAST, PANEL_FORECAST, PANEL_EXOGENOUS_FORECAST
from autosktime.data.benchmark.m4 import naive_2


def calculate_loss(
        solution: SUPPORTED_Y_TYPES,
        prediction: SUPPORTED_Y_TYPES,
        task_type: int,
        metric: BaseForecastingErrorMetric,
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

    if metric.get_tag("lower_is_better"):
        return score
    else:
        return -1 * score


def get_cost_of_crash(metric: BaseForecastingErrorMetric) -> float:
    if hasattr(metric, 'worst_score'):
        return metric.worst_score
    elif metric.get_tag("lower_is_better"):
        return MAXINT
    else:
        return 0


class MeanAbsolutePercentageError(MeanAbsolutePercentageError_):

    def __init__(self, symmetric: bool = True):
        super().__init__(symmetric=symmetric)
        self.worst_score = 1

    def __call__(
            self,
            y_true: SUPPORTED_Y_TYPES,
            y_pred: SUPPORTED_Y_TYPES,
            horizon_weight: Optional[np.ndarray] = None,
            **kwargs
    ) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            return super().__call__(y_true, y_pred, horizon_weight=horizon_weight, **kwargs)


class MedianAbsolutePercentageError(MedianAbsolutePercentageError_):

    def __call__(
            self,
            y_true: SUPPORTED_Y_TYPES,
            y_pred: SUPPORTED_Y_TYPES,
            horizon_weight: Optional[np.ndarray] = None,
            **kwargs
    ) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            return super().__call__(y_true, y_pred, horizon_weight=horizon_weight, **kwargs)


class MeanAbsoluteScaledError(MeanAbsoluteScaledError_):

    def __call__(
            self,
            y_true: SUPPORTED_Y_TYPES,
            y_pred: SUPPORTED_Y_TYPES,
            horizon_weight: Optional[np.ndarray] = None,
            **kwargs
    ) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            return super().__call__(y_true, y_pred, horizon_weight=horizon_weight, **kwargs)


class OverallWeightedAverage(_BaseForecastingErrorMetric):

    def __init__(self, multioutput="uniform_average", symmetric=True):
        name = "OverallWeightedAverage"
        func = overall_weighted_average
        super().__init__(
            func=func,
            name=name,
            multioutput=multioutput
        )


def overall_weighted_average(
        y_true: SUPPORTED_Y_TYPES,
        y_pred: SUPPORTED_Y_TYPES,
        horizon_weight: Optional[np.ndarray] = None,
        y_train: SUPPORTED_Y_TYPES = None,
        **kwargs
) -> float:
    if y_train is None:
        raise ValueError('y_train has to be provided for OWA')

    y_naive2 = naive_2(y_train)

    sMAPE = MeanAbsolutePercentageError()
    MASE = MeanAbsoluteScaledError()

    return (
                   sMAPE(y_true, y_pred, horizon_weight) / sMAPE(y_true, y_naive2, horizon_weight) +
                   MASE(y_true, y_pred, y_train=y_train) / MASE(y_true, y_naive2, horizon_weight, y_train=y_train)
           ) / 2


default_metric_for_task: Dict[int, BaseForecastingErrorMetric] = {
    UNIVARIATE_FORECAST: MeanAbsolutePercentageError(),
    UNIVARIATE_EXOGENOUS_FORECAST: MeanAbsolutePercentageError(),
    MULTIVARIATE_FORECAST: MeanAbsolutePercentageError(),
    MULTIVARIATE_EXOGENOUS_FORECAST: MeanAbsolutePercentageError(),
    PANEL_FORECAST: MeanAbsolutePercentageError(),
    PANEL_EXOGENOUS_FORECAST: MeanAbsolutePercentageError(),
}

STRING_TO_METRIC = {
    'mape': MeanAbsolutePercentageError(),
    'mdape': MedianAbsolutePercentageError(),
    'mase': MeanAbsoluteScaledError(),
    'owa': OverallWeightedAverage(),
}
METRIC_TO_STRING = {type(value): key for key, value in STRING_TO_METRIC.items()}
