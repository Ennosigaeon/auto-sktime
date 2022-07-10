# noinspection PyProtectedMember
import warnings
from typing import Optional, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics._regression import _check_reg_targets
from sklearn.utils import check_consistent_length
from sktime.performance_metrics.forecasting import (
    MeanAbsolutePercentageError as MeanAbsolutePercentageError_,
    MedianAbsolutePercentageError as MedianAbsolutePercentageError_,
    MeanAbsoluteScaledError as MeanAbsoluteScaledError_
)
# noinspection PyProtectedMember
from sktime.performance_metrics.forecasting._classes import BaseForecastingErrorMetric

from autosktime.constants import FORECAST_TASK, UNIVARIATE_FORECAST, UNIVARIATE_EXOGENOUS_FORECAST, MAXINT, \
    SUPPORTED_Y_TYPES, MULTIVARIATE_FORECAST, MULTIVARIATE_EXOGENOUS_FORECAST, PANEL_FORECAST, PANEL_EXOGENOUS_FORECAST
from autosktime.data.benchmark.m4 import naive_2


def calculate_loss(
        solution: SUPPORTED_Y_TYPES,
        prediction: SUPPORTED_Y_TYPES,
        task_type: int,
        metric: BaseForecastingErrorMetric,
        **kwargs
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
        score = metric(solution, prediction, **kwargs)
    else:
        raise NotImplementedError('Scoring of non FORECAST_TASK not supported')

    if metric.get_tag('lower_is_better'):
        return score
    else:
        return -1 * score


def get_cost_of_crash(metric: BaseForecastingErrorMetric) -> float:
    if hasattr(metric, 'worst_score'):
        return metric.worst_score
    elif metric.get_tag('lower_is_better'):
        return MAXINT
    else:
        return 0


class RootMeanSquaredError(BaseForecastingErrorMetric):
    def _evaluate(self, y_true, y_pred, **kwargs):
        return mean_squared_error(y_true, y_pred, multioutput=self.multioutput, squared=False)


class WeightedRootMeanSquaredError(BaseForecastingErrorMetric):
    def _evaluate(self, y_true, y_pred, **kwargs):
        # TODO how is this number known?
        t_j_eol = 1
        sample_weights = np.power(np.arange(y_true.shape[0]) / t_j_eol, 3)

        return mean_squared_error(y_true, y_pred, sample_weight=sample_weights, multioutput=self.multioutput,
                                  squared=False)


class MeanAbsoluteError(BaseForecastingErrorMetric):
    def _evaluate(self, y_true, y_pred, **kwargs):
        return mean_absolute_error(y_true, y_pred, multioutput=self.multioutput)


class WeightedMeanAbsoluteError(BaseForecastingErrorMetric):
    def _evaluate(self, y_true, y_pred, **kwargs):
        sample_weights = np.power(np.arange(y_true.shape[0]) / y_true.shape[0], 3)

        return mean_absolute_error(y_true, y_pred, sample_weight=sample_weights, multioutput=self.multioutput)


class MeanError(BaseForecastingErrorMetric):
    def _evaluate(self, y_true, y_pred, **kwargs):
        sample_weight = kwargs.get('sample_weight', None)

        y_type, y_true, y_pred, multioutput = _check_reg_targets(
            y_true, y_pred, self.multioutput
        )
        check_consistent_length(y_true, y_pred, sample_weight)
        output_errors = np.average(y_pred - y_true, weights=sample_weight, axis=0)
        if isinstance(multioutput, str):
            if multioutput == 'raw_values':
                return output_errors
            elif multioutput == 'uniform_average':
                # pass None as weights to np.average: uniform mean
                multioutput = None

        return np.average(output_errors, weights=multioutput)


class StandardDeviationError(BaseForecastingErrorMetric):
    def _evaluate(self, y_true, y_pred, **kwargs):
        diff = np.abs(y_true - y_pred)
        return diff.std().iloc[0]


class MeanArctangentAbsoluteRelativeError(BaseForecastingErrorMetric):
    def _evaluate(self, y_true, y_pred, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            y_type, y_true, y_pred, multioutput = _check_reg_targets(
                y_true, y_pred, self.multioutput
            )
            check_consistent_length(y_true, y_pred)
            output_errors = np.arctan(np.abs(y_true - y_pred) / y_true)

            if isinstance(multioutput, str):
                if multioutput == 'raw_values':
                    return output_errors
                elif multioutput == 'uniform_average':
                    # pass None as weights to np.average: uniform mean
                    multioutput = None

            return np.average(output_errors, weights=multioutput)


class RelativePrognosticHorizon(BaseForecastingErrorMetric):
    _tags = {
        'lower_is_better': False,
    }

    def __init__(self, alpha: float = 0.05):
        super().__init__()
        self.alpha = alpha

    def _evaluate(self, y_true, y_pred, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            upper_bound = y_true + self.alpha * y_true.shape[0]
            lower_bound = y_true - self.alpha * y_true.shape[0]

            in_bound = (lower_bound <= y_pred) & (y_pred <= upper_bound)

            t_j_alpha = self.find_longest_suffix(in_bound)

            return (y_true.shape[0] - t_j_alpha) / y_true.shape[0]

    def find_longest_suffix(self, x: pd.Series):
        if not np.all(x.iloc[-1]):
            return x.shape[0]

        # find run starts
        loc_run_start = np.empty(x.shape, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, x.shape[0]))

        return x.shape[0] - run_lengths[-1]


class PrognosticHorizonRate(BaseForecastingErrorMetric):
    _tags = {
        'lower_is_better': False,
    }

    def __init__(self, alpha: float = 0.05):
        super().__init__()
        self.alpha = alpha

    def _evaluate(self, y_true, y_pred, **kwargs):
        upper_bound = y_true + self.alpha * y_true.shape[0]
        lower_bound = y_true - self.alpha * y_true.shape[0]

        in_bound = (lower_bound <= y_pred) & (y_pred <= upper_bound)

        return (in_bound.sum() / in_bound.shape[0]).iloc[0]


class CumulativeRelativeAccuracy(BaseForecastingErrorMetric):
    _tags = {
        'lower_is_better': False,
    }

    def _evaluate(self, y_true, y_pred, **kwargs):
        return (1 - np.abs(y_true - y_pred) / y_true).mean().iloc[0]


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


class OverallWeightedAverage(BaseForecastingErrorMetric):

    def _evaluate(
            self,
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
    'rmse': RootMeanSquaredError(),
    'wrmse': WeightedRootMeanSquaredError(),
    'mae': MeanAbsoluteError(),
    'wmae': WeightedMeanAbsoluteError(),
    'me': MeanError(),
    'std': StandardDeviationError(),
    'maare': MeanArctangentAbsoluteRelativeError(),
    'relph': RelativePrognosticHorizon(),
    'phrate': PrognosticHorizonRate(),
    'cra': CumulativeRelativeAccuracy(),

    'mape': MeanAbsolutePercentageError(),
    'mdape': MedianAbsolutePercentageError(),
    'mase': MeanAbsoluteScaledError(),
    'owa': OverallWeightedAverage(),

}
METRIC_TO_STRING = {type(value): key for key, value in STRING_TO_METRIC.items()}
