import abc
import logging
import time
import warnings
from typing import Optional, Union, Type, TextIO

import numpy as np
import pandas as pd
from ConfigSpace import Configuration
from sktime.forecasting.base import ForecastingHorizon
# noinspection PyProtectedMember
from sktime.forecasting.model_selection._split import BaseSplitter, SingleWindowSplitter
from smac.tae import StatusType

from autosktime.automl_common.common.utils.backend import Backend
from autosktime.constants import FORECAST_TASK
from autosktime.data import AbstractDataManager
from autosktime.evaluation import TaFuncResult
from autosktime.metrics import Scorer, calculate_loss

__all__ = [
    'AbstractEvaluator'
]

from autosktime.pipeline.components.base import AutoSktimeComponent, AutoSktimePredictor
from autosktime.pipeline.components.forecast import ForecasterChoice


def _fit_and_suppress_warnings(
        logger: logging.Logger,
        model: AutoSktimeComponent,
        y: pd.Series
) -> AutoSktimeComponent:
    def send_warnings_to_log(
            message: Union[Warning, str],
            category: Type[Warning],
            filename: str,
            lineno: int,
            file: Optional[TextIO] = None,
            line: Optional[str] = None,
    ) -> None:
        logger.debug('{}:{}: {}:{}'.format(filename, lineno, str(category), message))

    with warnings.catch_warnings():
        warnings.showwarning = send_warnings_to_log
        model.fit(y)

    return model


class AbstractEvaluator:
    def __init__(
            self,
            backend: Backend,
            metric: Scorer,
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
            self.model = self._get_model()
            self.predict_function = self._predict_forecast
        else:
            raise ValueError('Unknown task_type {}'.format(self.task_type))

        self.logger = logging.getLogger(__name__)

        self.budget = budget
        self.budget_type = budget_type
        self.starttime = time.time()

    def _predict_forecast(self, fh: ForecastingHorizon, model: AutoSktimePredictor) -> pd.Series:
        def send_warnings_to_log(
                message: Union[Warning, str],
                category: Type[Warning],
                filename: str,
                lineno: int,
                file: Optional[TextIO] = None,
                line: Optional[str] = None,
        ) -> None:
            self.logger.debug('{}:{}: {}:{}'.format(filename, lineno, str(category), message))

        with warnings.catch_warnings():
            warnings.showwarning = send_warnings_to_log
            return model.predict(fh)

    @abc.abstractmethod
    def fit_predict_and_loss(self) -> None:
        """Fit, predict and compute the loss for cross-validation and
        holdout (both iterative and non-iterative)"""
        raise NotImplementedError()

    def get_splitter(self) -> BaseSplitter:
        # TODO not configurable
        test_size = 30
        return SingleWindowSplitter(np.arange(0, test_size) + 1)

    def _get_model(self) -> AutoSktimePredictor:
        # TODO not configurable
        return ForecasterChoice().set_hyperparameters(self.configuration)

    def _loss(self, y_true: pd.Series, y_hat: pd.Series) -> float:
        return calculate_loss(y_true, y_hat, self.task_type, self.metric)

    def finish_up(
            self,
            loss: float,
            train_loss: float,
            y_pred: pd.Series,
            status: StatusType,
    ) -> TaFuncResult:
        self.file_output(y_pred)

        additional_run_info = {
            'test_loss': {self.metric.name: loss},
            'train_loss': {self.metric.name: train_loss},
            'seed': self.seed,
            'duration': time.time() - self.starttime,
            'num_run': self.num_run
        }

        return TaFuncResult(loss=loss, additional_run_info=additional_run_info, status=status)

    def file_output(self, y_pred: pd.Series) -> None:
        self.backend.save_numrun_to_dir(
            seed=self.seed,
            idx=self.num_run,
            budget=self.budget,
            model=self.model,  # TODO type does not match
            cv_model=self.models if hasattr(self, 'models') else None,
            test_predictions=y_pred.values,
            valid_predictions=None,
            ensemble_predictions=None
        )
