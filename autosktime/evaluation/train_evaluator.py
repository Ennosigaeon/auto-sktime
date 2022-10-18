import glob
import os
from typing import Optional, Tuple, List, NamedTuple

import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon
# noinspection PyProtectedMember
from sktime.performance_metrics.forecasting._classes import BaseForecastingErrorMetric

from ConfigSpace import Configuration
from autosktime.constants import SUPPORTED_Y_TYPES
from autosktime.data.splitter import BaseSplitter
from autosktime.evaluation import TaFuncResult
# noinspection PyProtectedMember
from autosktime.evaluation.abstract_evaluator import AbstractEvaluator, _fit_and_suppress_warnings
from autosktime.pipeline.components.base import AutoSktimePredictor
from autosktime.pipeline.templates import TemplateChoice
from autosktime.util.backend import Backend
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
            budget: Optional[float] = None,
            budget_type: Optional[str] = None,
            verbose: bool = False,
    ):
        super().__init__(backend, metric, configuration, seed, random_state, num_run, budget, budget_type, verbose)
        self.splitter = splitter

        self.models: List[TemplateChoice] = []
        self.indices: List[Tuple[pd.Index, pd.Index]] = []

    def fit_predict_and_loss(self) -> TaFuncResult:
        y = self.datamanager.y

        n = self.splitter.get_n_splits(y)
        self.models = self._get_resampling_models(n)
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
                fit_and_predict = self._fit_and_predict_fold_standard
            elif self.budget_type == 'iterations' and self.models[i].supports_iterative_fit():
                fit_and_predict = self._fit_and_predict_fold_iterative
            else:
                fit_and_predict = self._fit_and_predict_fold_budget

            train_pred, test_pred = fit_and_predict(i, train=train_split, test=test_split)

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

            if self.verbose:
                self._log_progress(train_losses[i], test_loss[i],
                                   y.iloc[train_split], train_pred,
                                   y.iloc[test_split], test_pred,
                                   plot=False)

            test_weights[i] = len(test_split)

        train_weights /= train_weights.sum()
        test_weights /= test_weights.sum()

        train_loss = np.average(train_losses, weights=train_weights)
        test_loss = np.average(test_loss, weights=test_loss)

        # Retrain model on complete data
        self.model, y_ens = self._fit_and_predict_final_model()

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

        model = self.models[fold]
        _fit_and_suppress_warnings(self.logger, self.configuration.config_id, model, y_train, X_train, fh=None)

        self.indices[fold] = (y_train.index, y_test.index)

        train_pred = self.predict_function(y_train, X_train, model)
        test_pred = self.predict_function(y_test, X_test, model)

        return train_pred, test_pred

    def _fit_and_predict_fold_iterative(
            self,
            fold: int,
            train: np.ndarray,
            test: np.ndarray,
    ) -> Tuple[SUPPORTED_Y_TYPES, SUPPORTED_Y_TYPES]:
        model = self.models[fold]
        n_iter = int(np.ceil(self.budget / 100 * model.get_max_iter()))
        self.config_context.set_config(self.configuration.config_id, key='iterations', value=n_iter)
        model.budget = self.budget

        return self._fit_and_predict_fold_standard(fold, train, test)

    def _fit_and_predict_fold_budget(
            self,
            fold: int,
            train: np.ndarray,
            test: np.ndarray,
    ) -> Tuple[SUPPORTED_Y_TYPES, SUPPORTED_Y_TYPES]:
        raise ValueError('Budgets not supported yet')

    def _fit_and_predict_final_model(
            self,
            fh: ForecastingHorizon = None,
            refit: bool = False,
    ) -> Tuple[AutoSktimePredictor, SUPPORTED_Y_TYPES]:
        if refit:
            y = self.datamanager.y
            X = self.datamanager.X

            model = self._get_model()

            if self.budget_type == 'iterations' and model.supports_iterative_fit():
                n_iter = int(np.ceil(self.budget / 100 * model.get_max_iter()))
                self.config_context.set_config(self.configuration.config_id, key='iterations', value=n_iter)

            _fit_and_suppress_warnings(self.logger, self.configuration.config_id, model, y, X, fh)
        else:
            model = self.models[0]

        test_pred = self.predict_function(self.datamanager.y_ens, self.datamanager.X_ens, model)

        return model, test_pred


class MultiFidelityTrainEvaluator(TrainEvaluator):

    def _get_model(self) -> AutoSktimePredictor:
        budget = self.__get_previous_budget()
        if budget is None:
            return super()._get_model()
        else:
            model = self.backend.load_model_by_seed_and_id_and_budget(self.seed, self.num_run, budget)
            # noinspection PyTypeChecker
            return model

    def _get_resampling_models(self, n: int):
        budget = self.__get_previous_budget()
        if budget is None:
            return super()._get_resampling_models(n)
        else:
            resampling_models = self.backend.load_cv_model_by_seed_and_id_and_budget(self.seed, self.num_run, budget)
            return resampling_models

    def __get_previous_budget(self) -> Optional[float]:
        seed = self.seed
        idx = self.num_run
        prefix = os.path.join(self.backend.get_runs_directory(), f'{seed}_{idx}_')
        previous_budgets = [float(path[len(prefix):]) for path in glob.glob(f'{prefix}*')]
        return max(previous_budgets, default=None)


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
    instance = MultiFidelityTrainEvaluator if budget_type == 'iterations' else TrainEvaluator
    evaluator = instance(
        backend=backend,
        metric=metric,
        configuration=config,
        splitter=splitter,
        seed=seed,
        random_state=random_state,
        num_run=num_run,
        budget=budget,
        budget_type=budget_type,
        verbose=verbose,
    )
    result = evaluator.fit_predict_and_loss()
    return result
