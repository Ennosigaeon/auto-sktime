import warnings
from typing import Optional, Dict

import numpy as np

from autosktime.constants import FORECAST_TASK, UNIVARIATE_FORECAST, UNIVARIATE_EXOGENOUS_FORECAST, MAXINT
from autosktime.data.benchmark.m4 import naive_2
from sktime.forecasting.model_selection._split import ACCEPTED_Y_TYPES
from sktime.performance_metrics.forecasting import (
    MeanAbsolutePercentageError as MeanAbsolutePercentageError_,
    MedianAbsolutePercentageError as MedianAbsolutePercentageError_,
    MeanAbsoluteScaledError as MeanAbsoluteScaledError_
)
# noinspection PyProtectedMember
from sktime.performance_metrics.forecasting._classes import _BaseForecastingErrorMetric, \
    _PercentageForecastingErrorMetric

BaseMetric = _BaseForecastingErrorMetric


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
        return -1 * score
    else:
        return score


def get_cost_of_crash(metric: BaseMetric) -> float:
    if isinstance(metric, _PercentageForecastingErrorMetric):
        return 1
    else:
        return MAXINT


class MeanAbsolutePercentageError(MeanAbsolutePercentageError_):

    def __init__(self):
        super().__init__(symmetric=True)

    def __call__(
            self,
            y_true: ACCEPTED_Y_TYPES,
            y_pred: ACCEPTED_Y_TYPES,
            horizon_weight: Optional[np.ndarray] = None,
            **kwargs
    ) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            return super().__call__(y_true, y_pred, horizon_weight=horizon_weight, **kwargs)


class MedianAbsolutePercentageError(MedianAbsolutePercentageError_):

    def __init__(self):
        super().__init__()

    def __call__(
            self,
            y_true: ACCEPTED_Y_TYPES,
            y_pred: ACCEPTED_Y_TYPES,
            horizon_weight: Optional[np.ndarray] = None,
            **kwargs
    ) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            return super().__call__(y_true, y_pred, horizon_weight=horizon_weight, **kwargs)


class MeanAbsoluteScaledError(MeanAbsoluteScaledError_):

    def __call__(
            self,
            y_true: ACCEPTED_Y_TYPES,
            y_pred: ACCEPTED_Y_TYPES,
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
        y_true: ACCEPTED_Y_TYPES,
        y_pred: ACCEPTED_Y_TYPES,
        horizon_weight: Optional[np.ndarray] = None,
        y_train: ACCEPTED_Y_TYPES = None,
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


default_metric_for_task: Dict[int, BaseMetric] = {
    UNIVARIATE_FORECAST: MeanAbsolutePercentageError(),
    UNIVARIATE_EXOGENOUS_FORECAST: MeanAbsolutePercentageError(),
}

STRING_TO_METRIC = {
    'mape': MeanAbsolutePercentageError(),
    'mdape': MedianAbsolutePercentageError(),
    'mase': MeanAbsoluteScaledError(),
    'owa': OverallWeightedAverage(),
}
METRIC_TO_STRING = {value: key for key, value in STRING_TO_METRIC.items()}
