import abc
import logging
import time
import warnings
from typing import Optional, Union, Type, TextIO

import pandas as pd
from sktime.performance_metrics.forecasting._classes import BaseForecastingErrorMetric

from smac.tae import StatusType

from ConfigSpace import Configuration
from autosktime.automl_common.common.utils.backend import Backend
from autosktime.constants import FORECAST_TASK
from autosktime.data import AbstractDataManager
from autosktime.evaluation import TaFuncResult
from autosktime.metrics import calculate_loss, get_cost_of_crash
from autosktime.pipeline.templates.univariate_endogenous import UnivariateEndogenousPipeline
from sktime.forecasting.base import ForecastingHorizon

__all__ = [
    'AbstractEvaluator'
]

from autosktime.pipeline.components.base import AutoSktimeComponent, AutoSktimePredictor


def _fit_and_suppress_warnings(
        logger: logging.Logger,
        model: AutoSktimeComponent,
        y: pd.Series,
        X: Optional[pd.DataFrame]
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
        model.fit(y, X=X)

    return model


class AbstractEvaluator:
    def __init__(
            self,
            backend: Backend,
            metric: BaseForecastingErrorMetric,
            configuration: Configuration,
            seed: int = 1,
            num_run: int = 0,
            budget: Optional[float] = None,
            budget_type: Optional[str] = None,
    ):
        self.configuration = configuration
        self.backend = backend

        self.datamanager: AbstractDataManager = self.backend.load_datamanager()

        self.metric = metric
        self.task_type = self.datamanager.info['task']
        self.seed = seed
        self.num_run = num_run

        if self.task_type in FORECAST_TASK:
            self.model: Optional[AutoSktimePredictor] = None
            self.predict_function = self._predict_forecast
        else:
            raise ValueError(f'Unknown task_type {self.task_type}')

        self.logger = logging.getLogger(__name__)

        self.budget = budget
        self.budget_type = budget_type
        self.starttime = time.time()

    def _predict_forecast(
            self,
            fh: ForecastingHorizon,
            X: Optional[pd.DataFrame],
            model: AutoSktimePredictor,
            name: str
    ) -> pd.Series:
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
            y_pred = model.predict(fh, X=X)
            y_pred.name = name
            return y_pred

    @abc.abstractmethod
    def fit_predict_and_loss(self) -> None:
        """Fit, predict and compute the loss for cross-validation and
        holdout (both iterative and non-iterative)"""
        raise NotImplementedError()

    def _get_model(self) -> AutoSktimePredictor:
        # TODO not configurable
        return UnivariateEndogenousPipeline(
            config=self.configuration,
            dataset_properties=self.datamanager.dataset_properties)

    def _loss(self, y_true: pd.Series, y_hat: pd.Series, error: str = 'raise') -> float:
        try:
            return calculate_loss(y_true, y_hat, self.task_type, self.metric)
        except ValueError:
            if error == 'raise':
                raise
            elif error == 'worst':
                return get_cost_of_crash(self.metric)
            else:
                raise ValueError(f"Unknown exception handling '{error}' method")

    def finish_up(
            self,
            loss: float,
            train_loss: float,
            y_pred: pd.Series,
            y_ens: pd.Series,
            status: StatusType,
    ) -> TaFuncResult:
        self.file_output(y_pred, y_ens)

        additional_run_info = {
            'test_loss': {self.metric.name: loss},
            'train_loss': {self.metric.name: train_loss},
            'seed': self.seed,
            'duration': time.time() - self.starttime,
            'num_run': self.num_run
        }

        return TaFuncResult(loss=loss, additional_run_info=additional_run_info, status=status)

    def file_output(self, y_pred: pd.Series, y_ens: pd.Series) -> None:
        # noinspection PyTypeChecker
        self.backend.save_numrun_to_dir(
            seed=self.seed,
            idx=self.num_run,
            budget=self.budget,
            model=self.model,  # TODO type does not match
            cv_model=self.models if hasattr(self, 'models') else None,
            test_predictions=y_pred,
            valid_predictions=None,
            ensemble_predictions=y_ens
        )
