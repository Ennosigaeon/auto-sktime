import os.path
from typing import Optional

import numpy as np
import pandas as pd
from ConfigSpace import Configuration
# noinspection PyProtectedMember
from sktime.performance_metrics.forecasting._classes import BaseForecastingErrorMetric

from autosktime.constants import Budget
from autosktime.data.splitter import BaseSplitter
from autosktime.evaluation import TaFuncResult
# noinspection PyProtectedMember
from autosktime.evaluation.abstract_evaluator import AbstractEvaluator, _fit_and_suppress_warnings
from autosktime.util.backend import Backend


class TestEvaluator(AbstractEvaluator):

    def fit_predict_and_loss(self) -> TaFuncResult:
        y = self.datamanager.y
        X = self.datamanager.X

        self.model = self._get_model()

        if self.budget_type is None:
            fit_and_predict = self._fit_and_predict_standard
        elif self.budget_type == Budget.Iterations and self.model.supports_iterative_fit():
            fit_and_predict = self._fit_and_predict_iterative
        elif self.budget_type == Budget.SeriesLength:
            fit_and_predict = self._fit_and_predict_standard
        else:
            fit_and_predict = self._fit_and_predict_standard

        test_pred = fit_and_predict(y, X)
        if pd.isna(test_pred).any(axis=None):
            test_pred = y
        loss = self._loss(y, test_pred, error='worst')

        if os.path.exists(self.backend.get_numrun_directory(self.seed, self.num_run, self.budget)):
            os.rename(
                self.backend.get_numrun_directory(self.seed, self.num_run, self.budget),
                self.backend.get_numrun_directory(self.seed, self.num_run, self.budget) + '_partial',
            )

        return self.finish_up(
            loss=loss,
            train_loss=loss,
            test_loss=loss,
            y_pred=test_pred,
            y_ens=test_pred,
            y_test=test_pred,
        )

    def _fit_and_predict_standard(self, y: pd.Series, X: pd.DataFrame):
        _fit_and_suppress_warnings(self.logger, self.configuration.config_id, self.model, y, X, fh=None)
        test_pred = self.predict_function(self.datamanager.y_ens, self.datamanager.X_ens, self.model)
        return test_pred

    def _fit_and_predict_iterative(self, y: pd.Series, X: pd.DataFrame):
        n_iter = int(np.ceil(self.budget / 100 * self.model.get_max_iter()))
        self.config_context.set_config(self.configuration.config_id, key='iterations', value=n_iter)
        return self._fit_and_predict_standard(y, X)


def evaluate(
        config: Configuration,
        backend: Backend,
        metric: BaseForecastingErrorMetric,
        seed: int,
        random_state: np.random.RandomState,
        num_run: int,
        splitter: BaseSplitter,
        refit: bool = False,
        budget: Optional[float] = 100.0,
        budget_type: Optional[Budget] = None,
        verbose: bool = False,
) -> TaFuncResult:
    evaluator = TestEvaluator(
        backend=backend,
        metric=metric,
        configuration=config,
        seed=seed,
        random_state=random_state,
        num_run=num_run,
        budget=budget,
        budget_type=budget_type,
        verbose=verbose,
    )
    result = evaluator.fit_predict_and_loss()
    return result
