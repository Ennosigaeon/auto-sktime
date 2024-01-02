import abc
import logging
import time
import warnings
from typing import Optional, Union, Type, TextIO, List, Hashable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sktime.forecasting.base import ForecastingHorizon
from sktime.performance_metrics.forecasting._classes import BaseForecastingErrorMetric

from ConfigSpace import Configuration
from autosktime.automl_common.common.utils.backend import Backend
from autosktime.constants import FORECAST_TASK, SUPPORTED_Y_TYPES, Budget
from autosktime.data import DataManager
from autosktime.evaluation import TaFuncResult
from autosktime.metrics import calculate_loss, get_cost_of_crash
from autosktime.pipeline.templates import TemplateChoice
from autosktime.util import resolve_index
from autosktime.util.backend import ConfigContext, ConfigId
from autosktime.util.plotting import plot_grouped_series

__all__ = [
    'AbstractEvaluator'
]

from autosktime.pipeline.components.base import AutoSktimeComponent, AutoSktimePredictor


def _fit_and_suppress_warnings(
        logger: logging.Logger,
        config_id: ConfigId,
        model: AutoSktimeComponent,
        y: SUPPORTED_Y_TYPES,
        X: Optional[pd.DataFrame] = None,
        fh: Optional[ForecastingHorizon] = None
) -> AutoSktimeComponent:
    def send_warnings_to_log(
            message: Union[Warning, str],
            category: Type[Warning],
            filename: str,
            lineno: int,
            file: Optional[TextIO] = None,
            line: Optional[str] = None,
    ) -> None:
        logger.debug(f'{filename}:{lineno}: {str(category)}:{message}')

    with warnings.catch_warnings():
        warnings.showwarning = send_warnings_to_log
        model.set_config_id(config_id)
        config: ConfigContext = ConfigContext.instance()

        config.set_config(config_id, key='is_fitting', value=True)
        if model.is_fitted:
            # noinspection PyUnresolvedReferences
            model.update(y, X=X)
        else:
            # noinspection PyUnresolvedReferences
            model.fit(y, X=X, fh=fh)
        config.reset_config(config_id, key='is_fitting')

    return model


class AbstractEvaluator:
    def __init__(
            self,
            backend: Backend,
            metric: BaseForecastingErrorMetric,
            configuration: Configuration,
            seed: int = 1,
            random_state: np.random.RandomState = None,
            num_run: int = 0,
            budget: Optional[float] = None,
            budget_type: Optional[Budget] = None,
            verbose: bool = False
    ):
        self.configuration = configuration
        self.backend = backend
        self.config_context: ConfigContext = ConfigContext.instance()

        self.datamanager: DataManager = self.backend.load_datamanager()

        self.metric = metric
        self.task_type = self.datamanager.info['task']
        self.seed = seed
        self.random_state = random_state
        self.num_run = num_run

        if self.task_type in FORECAST_TASK:
            self.model: Optional[AutoSktimePredictor] = None
            self.predict_function = self._predict_forecast
        else:
            raise ValueError(f'Unknown task_type {self.task_type}')

        self.logger = logging.getLogger(__name__)

        self.budget = budget
        self.budget_type = budget_type
        self.verbose = verbose
        self.starttime = time.time()

    def _predict_forecast(
            self,
            y: SUPPORTED_Y_TYPES,
            X: Optional[pd.DataFrame],
            model: AutoSktimePredictor
    ) -> SUPPORTED_Y_TYPES:
        def send_warnings_to_log(
                message: Union[Warning, str],
                category: Type[Warning],
                filename: str,
                lineno: int,
                file: Optional[TextIO] = None,
                line: Optional[str] = None,
        ) -> None:
            self.logger.debug(f'{filename}:{lineno}: {str(category)}:{message}')

        with warnings.catch_warnings():
            warnings.showwarning = send_warnings_to_log

            fh = ForecastingHorizon(resolve_index(y.index), is_relative=False)
            name = self._get_name(y)

            y_pred = model.predict(fh, X=X)
            y_pred.name = name

            assert y_pred.shape == y.shape, 'Shape of predictions does not match input data. This should never happen'
            return y_pred

    @staticmethod
    def _get_name(y: SUPPORTED_Y_TYPES) -> Hashable:
        if isinstance(y, pd.Series):
            return y.name
        elif isinstance(y, pd.DataFrame) and len(y.columns) == 1:
            return y.columns[0]
        return ''

    @abc.abstractmethod
    def fit_predict_and_loss(self) -> None:
        """Fit, predict and compute the loss for cross-validation and
        holdout (both iterative and non-iterative)"""
        raise NotImplementedError()

    def _get_model(self) -> TemplateChoice:
        return TemplateChoice.from_config(self.configuration, self.budget, self.datamanager.dataset_properties,
                                          self.random_state)

    def _get_resampling_models(self, n: int) -> List[TemplateChoice]:
        return [self._get_model() for _ in range(n)]

    def _loss(self, y_true: SUPPORTED_Y_TYPES, y_hat: SUPPORTED_Y_TYPES, error: str = 'raise', **kwargs) -> float:
        try:
            return calculate_loss(y_true, y_hat, self.task_type, self.metric, **kwargs)
        except (ValueError, TypeError):
            if error == 'raise':
                raise
            elif error == 'worst':
                return get_cost_of_crash(self.metric)
            else:
                raise ValueError(f"Unknown exception handling '{error}' method")

    def _log_progress(self, train_loss: float, val_loss: float, test_loss: float,
                      y_train: SUPPORTED_Y_TYPES, y_train_pred: SUPPORTED_Y_TYPES,
                      y_val: SUPPORTED_Y_TYPES, y_val_pred: SUPPORTED_Y_TYPES,
                      y_test: Optional[SUPPORTED_Y_TYPES], y_test_pred: Optional[SUPPORTED_Y_TYPES],
                      plot: bool = False):
        self.logger.debug(f'Finished fold with train loss {train_loss} and validation loss {val_loss}'
                          f'{f" and test loss {test_loss}" if y_test is not None else ""}')
        if plot:
            plot_grouped_series(None, y_val, y_val_pred)
            plt.show()
            plot_grouped_series(None, y_train, y_train_pred)
            plt.show()
            if y_test is not None:
                plot_grouped_series(None, y_test, y_test_pred)
                plt.show()

    def finish_up(
            self,
            loss: float,
            train_loss: float,
            test_loss: float,
            y_pred: SUPPORTED_Y_TYPES,
            y_ens: SUPPORTED_Y_TYPES,
            y_test: Optional[SUPPORTED_Y_TYPES],
    ) -> TaFuncResult:
        self.file_output(y_pred, y_ens, y_test)

        additional_run_info = {
            'val_loss': {self.metric.name: loss},
            'train_loss': {self.metric.name: train_loss},
            'test_loss': {self.metric.name: test_loss},
            'seed': self.seed,
            'duration': time.time() - self.starttime,
            'num_run': self.num_run
        }

        return loss, additional_run_info

    def file_output(
            self,
            y_pred: SUPPORTED_Y_TYPES,
            y_ens: SUPPORTED_Y_TYPES,
            y_test: Optional[SUPPORTED_Y_TYPES]
    ) -> None:
        # Results have to be stored as float32
        y_pred = y_pred.astype(dtype=np.float32)
        y_ens = y_ens.astype(dtype=np.float32)
        y_test = y_test.astype(dtype=np.float32) if y_test is not None else None

        # Fail-fast if any prediction contains illegal values
        if y_pred is not None and (np.isinf(y_pred).any().any() or np.isnan(y_pred).any().any()):
            raise ValueError('"y_pred" predictions contain nan, inf or values larger that float32')
        if y_ens is not None and (np.isinf(y_ens).any().any() or np.isnan(y_ens).any().any()):
            raise ValueError('"y_ens" predictions contain nan, inf or values larger that float32')
        if y_test is not None and (np.isinf(y_test).any().any() or np.isnan(y_test).any().any()):
            raise ValueError('"y_test" predictions contain nan, inf or values larger that float32')

        self.backend.save_numrun_to_dir(
            seed=self.seed,
            idx=self.num_run,
            budget=self.budget,
            model=self.model,
            cv_model=self.models if hasattr(self, 'models') else None,
            test_predictions=y_test,
            valid_predictions=y_pred,
            ensemble_predictions=y_ens
        )
