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
        y_test = self.datamanager.y_test

        n = self.splitter.get_n_splits(y)
        self.models = self._get_resampling_models(n)
        self.indices = [None] * n

        # noinspection PyTypeChecker
        train_losses = np.empty(n)
        val_loss = np.empty(n)
        test_loss = np.empty(n)

        train_weights = np.empty(n)
        val_weights = np.empty(n)

        # noinspection PyTypeChecker
        val_pred: pd.Series = None  # only for mypy
        for i, (train_split, val_split) in enumerate(self.splitter.split(y)):
            if self.budget_type is None:
                fit_and_predict = self._fit_and_predict_fold_standard
            elif self.budget_type == 'iterations' and self.models[i].supports_iterative_fit():
                fit_and_predict = self._fit_and_predict_fold_iterative
            else:
                fit_and_predict = self._fit_and_predict_fold_budget

            train_pred, val_pred, test_pred = fit_and_predict(i, train=train_split, val=val_split)

            # Compute train loss of this fold and store it. train_loss could
            # either be a scalar or a dict of scalars with metrics as keys.
            train_losses[i] = self._loss(
                y.iloc[train_split],
                train_pred,
                error='worst'
            )
            train_weights[i] = len(train_split)

            # Compute validation loss of this fold and store it.
            val_loss[i] = self._loss(
                y.iloc[val_split],
                val_pred,
                error='raise'
            )
            val_weights[i] = len(val_split)

            # Compute test loss if y_test is given
            test_loss[i] = self._loss(
                y_test,
                test_pred,
                error='worst'
            )

            if self.verbose:
                self._log_progress(train_losses[i], val_loss[i], test_loss[i],
                                   y.iloc[train_split], train_pred,
                                   y.iloc[val_split], val_pred,
                                   y_test, test_pred,
                                   plot=False)

        train_weights /= train_weights.sum()
        val_weights /= val_weights.sum()

        train_loss = np.average(train_losses, weights=train_weights)
        val_loss = np.average(val_loss, weights=val_weights)
        test_loss = np.average(test_loss)

        # Retrain model on complete data
        self.model, y_ens, y_test = self._fit_and_predict_final_model()

        return self.finish_up(
            loss=val_loss,
            train_loss=train_loss,
            test_loss=test_loss,
            y_pred=val_pred,
            y_ens=y_ens,
            y_test=y_test,
            status=StatusType.SUCCESS
        )

    def _fit_and_predict_fold_standard(
            self,
            fold: int,
            train: np.ndarray,
            val: np.ndarray
    ) -> Tuple[SUPPORTED_Y_TYPES, SUPPORTED_Y_TYPES, Optional[SUPPORTED_Y_TYPES]]:
        y = self.datamanager.y
        y_train = y.iloc[train]
        y_val = y.iloc[val]
        y_test = self.datamanager.y_test

        X = self.datamanager.X
        if X is None:
            X_train = None
            X_val = None
            X_test = None
        else:
            X_train = X.iloc[train, :]
            X_val = X.iloc[val, :]
            X_test = self.datamanager.X_test

        model = self.models[fold]
        _fit_and_suppress_warnings(self.logger, self.configuration.config_id, model, y_train, X_train, fh=None)

        self.indices[fold] = (y_train.index, y_val.index)

        train_pred = self.predict_function(y_train, X_train, model)
        val_pred = self.predict_function(y_val, X_val, model)
        test_pred = self.predict_function(y_test, X_test, model) if y_test is not None else None

        return train_pred, val_pred, test_pred

    def _fit_and_predict_fold_iterative(
            self,
            fold: int,
            train: np.ndarray,
            val: np.ndarray,
    ) -> Tuple[SUPPORTED_Y_TYPES, SUPPORTED_Y_TYPES, Optional[SUPPORTED_Y_TYPES]]:
        model = self.models[fold]
        n_iter = int(np.ceil(self.budget / 100 * model.get_max_iter()))
        self.config_context.set_config(self.configuration.config_id, key='iterations', value=n_iter)
        model.budget = self.budget

        return self._fit_and_predict_fold_standard(fold, train, val)

    def _fit_and_predict_fold_budget(
            self,
            fold: int,
            train: np.ndarray,
            val: np.ndarray,
    ) -> Tuple[SUPPORTED_Y_TYPES, SUPPORTED_Y_TYPES, Optional[SUPPORTED_Y_TYPES]]:
        raise ValueError('Budgets not supported yet')

    def _fit_and_predict_final_model(
            self,
            fh: ForecastingHorizon = None,
            refit: bool = False,
    ) -> Tuple[AutoSktimePredictor, SUPPORTED_Y_TYPES, Optional[SUPPORTED_Y_TYPES]]:
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

        ens_pred = self.predict_function(self.datamanager.y_ens, self.datamanager.X_ens, model)
        test_pred = self.predict_function(self.datamanager.y_test, self.datamanager.X_test, model) \
            if self.datamanager.y_test is not None else None

        return model, ens_pred, test_pred


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
