from typing import Optional

import numpy as np
# noinspection PyProtectedMember
from sktime.performance_metrics.forecasting._classes import BaseForecastingErrorMetric
from smac.tae import StatusType

from ConfigSpace import Configuration
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

        if self.model.budget != 0.0:
            n_iter = int(np.ceil(self.budget / 100 * self.model.get_max_iter()))
            self.model.set_desired_iterations(n_iter)

        _fit_and_suppress_warnings(self.logger, self.model, y, X, fh=None)
        test_pred = self.predict_function(self.datamanager.y_ens, self.datamanager.X_ens, self.model)
        loss = self._loss(y, test_pred, error='raise')

        return self.finish_up(
            loss=loss,
            train_loss=loss,
            y_pred=test_pred,
            y_ens=test_pred,
            status=StatusType.SUCCESS
        )


def evaluate(
        config: Configuration,
        backend: Backend,
        metric: BaseForecastingErrorMetric,
        seed: int,
        random_state: np.random.RandomState,
        num_run: int,
        splitter: BaseSplitter,
        budget: Optional[float] = 100.0,
        budget_type: Optional[str] = None,
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
