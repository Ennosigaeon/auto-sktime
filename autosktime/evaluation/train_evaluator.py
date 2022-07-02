from typing import Optional, Tuple, List, NamedTuple

import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon

from ConfigSpace import Configuration
from autosktime.constants import SUPPORTED_Y_TYPES
from autosktime.data.splitter import BaseSplitter
from autosktime.ensembles.util import get_ensemble_targets
from autosktime.evaluation import TaFuncResult
# noinspection PyProtectedMember
from autosktime.evaluation.abstract_evaluator import AbstractEvaluator, _fit_and_suppress_warnings
from autosktime.pipeline.components.base import AutoSktimeComponent, AutoSktimePredictor
from autosktime.util.backend import Backend
# noinspection PyProtectedMember
from sktime.performance_metrics.forecasting._classes import BaseForecastingErrorMetric

from smac.tae import StatusType

EvalResult = NamedTuple('EvalResult', [
    ('y_pred', pd.Series),
    ('loss', float),
    ('weight', float)
])

FoldResult = NamedTuple('FoldResult', [
    ('train', EvalResult),
    ('test', EvalResult),
])


class TrainEvaluator(AbstractEvaluator):

    def __init__(
            self,
            backend: Backend,
            metric: BaseForecastingErrorMetric,
            configuration: Configuration,
            splitter: BaseSplitter,
            seed: int = 1,
            random_state: np.random.RandomState = None,
            num_run: int = 0,
            ensemble_size: float = 0.2,
            budget: Optional[float] = None,
            budget_type: Optional[str] = None,
    ):
        super().__init__(backend, metric, configuration, seed, random_state, num_run, budget, budget_type)
        self.splitter = splitter
        self.ensemble_size = ensemble_size

        self.models: List[AutoSktimeComponent] = []
        self.indices: List[Tuple[pd.Index, pd.Index]] = []

    def fit_predict_and_loss(self) -> TaFuncResult:
        y = self.datamanager.y

        n = self.splitter.get_n_splits(y)
        self.models = [None] * n
        self.indices = [None] * n

        # noinspection PyTypeChecker
        train_losses = np.empty(n)
        test_loss = np.empty(n)

        train_weights = np.empty(n)
        test_weights = np.empty(n)

        # noinspection PyTypeChecker
        test_pred: pd.Series = None  # only for mypy
        for i, (train_split, test_split) in enumerate(self.splitter.split(y)):
            if self.budget_type is None:
                (
                    train_pred,
                    test_pred,
                ) = self._fit_and_predict_fold_standard(i, train=train_split, test=test_split)

            else:
                (
                    train_pred,
                    test_pred,
                ) = self._fit_and_predict_fold_budget(i, train=train_split, test=test_split)

            # Compute train loss of this fold and store it. train_loss could
            # either be a scalar or a dict of scalars with metrics as keys.
            train_losses[i] = self._loss(
                y.iloc[train_split],
                train_pred,
                error='worst'
            )
            train_weights[i] = len(train_split)

            # Compute validation loss of this fold and store it.
            test_loss[i] = self._loss(
                y.iloc[test_split],
                test_pred,
                error='raise'
            )
            test_weights[i] = len(test_split)

        train_weights /= train_weights.sum()
        test_weights /= test_weights.sum()

        train_loss = np.average(train_losses, weights=train_weights)
        test_loss = np.average(test_loss, weights=test_loss)

        # Retrain model on complete data
        # TODO is this really the correct place? Maybe only retrain well-performing models?
        self.model, y_ens = self._fit_and_predict_final_model(self.ensemble_size)

        return self.finish_up(
            loss=test_loss,
            train_loss=train_loss,
            y_pred=test_pred,
            y_ens=y_ens,
            status=StatusType.SUCCESS
        )

    def _fit_and_predict_fold_standard(
            self,
            fold: int,
            train: np.ndarray,
            test: np.ndarray
    ) -> Tuple[SUPPORTED_Y_TYPES, SUPPORTED_Y_TYPES]:
        y = self.datamanager.y
        y_train = y.iloc[train]
        y_test = y.iloc[test]

        X = self.datamanager.X
        if X is None:
            X_train = None
            X_test = None
        else:
            X_train = X.iloc[train, :]
            X_test = X.iloc[test, :]

        model = self._get_model()
        _fit_and_suppress_warnings(self.logger, model, y_train, X_train, fh=None)

        self.models[fold] = model
        self.indices[fold] = (y_train.index, y_test.index)

        train_pred = self.predict_function(y_train, X_train, model)
        test_pred = self.predict_function(y_test, X_test, model)

        return train_pred, test_pred

    def _fit_and_predict_fold_budget(
            self,
            fold: int,
            train: np.ndarray,
            test: np.ndarray,
    ) -> Tuple[SUPPORTED_Y_TYPES, SUPPORTED_Y_TYPES]:
        raise NotImplementedError('Budgets not supported yet')

    def _fit_and_predict_final_model(
            self,
            ensemble_size: float = 0.2,
            fh: ForecastingHorizon = None
    ) -> Tuple[AutoSktimePredictor, SUPPORTED_Y_TYPES]:
        y = self.datamanager.y
        X = self.datamanager.X

        model = self._get_model()
        _fit_and_suppress_warnings(self.logger, model, y, X, fh)
        self.model = model

        splitter = type(self.splitter)(fh=ensemble_size)
        splitter.random_state = 42

        y_test, X_test = get_ensemble_targets(self.datamanager, splitter)
        test_pred = self.predict_function(y_test, X_test, model)

        return model, test_pred


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
) -> TaFuncResult:
    evaluator = TrainEvaluator(
        backend=backend,
        metric=metric,
        configuration=config,
        splitter=splitter,
        seed=seed,
        random_state=random_state,
        num_run=num_run,
        budget=budget,
        budget_type=budget_type,
    )
    result = evaluator.fit_predict_and_loss()
    return result
