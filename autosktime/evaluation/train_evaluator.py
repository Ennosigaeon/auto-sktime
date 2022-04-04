from functools import partial
from typing import Optional, Tuple, List, Type, Dict, Any, NamedTuple

import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon
# noinspection PyProtectedMember
from sktime.forecasting.model_selection._split import BaseSplitter
from smac.tae import StatusType

from ConfigSpace import Configuration
from autosktime.automl_common.common.utils.backend import Backend
from autosktime.data.splitter import SingleWindowSplitter, SlidingWindowSplitter
from autosktime.evaluation import TaFuncResult
# noinspection PyProtectedMember
from autosktime.evaluation.abstract_evaluator import AbstractEvaluator, _fit_and_suppress_warnings
from autosktime.metrics import Scorer
from autosktime.pipeline.components.base import AutoSktimeComponent

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
            metric: Scorer,
            configuration: Configuration,
            splitter: Type[BaseSplitter],
            splitter_kwargs: Dict[str, Any] = None,
            seed: int = 1,
            num_run: int = 0,
            budget: Optional[float] = None,
            budget_type: Optional[str] = None,
    ):
        super().__init__(backend, metric, configuration, seed, num_run, budget, budget_type)
        self.splitter = self._get_splitter(splitter, splitter_kwargs)

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
                y[train_split],
                train_pred,
            )
            train_weights[i] = len(train_split)

            # Compute validation loss of this fold and store it.
            test_loss[i] = self._loss(
                y[test_split],
                test_pred,
            )
            test_weights[i] = len(test_split)

        train_weights /= train_weights.sum()
        test_weights /= test_weights.sum()

        train_loss = np.average(train_losses, weights=train_weights)
        test_loss = np.average(test_loss, weights=test_loss)

        return self.finish_up(loss=test_loss, train_loss=train_loss, y_pred=test_pred, status=StatusType.SUCCESS)

    def _fit_and_predict_fold_standard(
            self,
            fold: int,
            train: np.ndarray,
            test: np.ndarray
    ) -> Tuple[pd.Series, pd.Series]:
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
        _fit_and_suppress_warnings(self.logger, model, y_train, X_train)

        self.models[fold] = model
        self.indices[fold] = (y_train.index, y_test.index)

        train_pred = self.predict_function(ForecastingHorizon(y_train.index, is_relative=False), X_train, model)
        test_pred = self.predict_function(ForecastingHorizon(y_test.index, is_relative=False), X_test, model)

        return train_pred, test_pred

    def _fit_and_predict_fold_budget(
            self,
            fold: int,
            train: np.ndarray,
            test: np.ndarray,
    ) -> Tuple[pd.Series, pd.Series]:
        raise NotImplementedError('Budgets not supported yet')


def _eval(
        config: Configuration,
        backend: Backend,
        metric: Scorer,
        seed: int,
        num_run: int,
        splitter: Type[BaseSplitter],
        budget: Optional[float] = 100.0,
        budget_type: Optional[str] = None,
        resampling_strategy_args: Dict[str, Any] = None,
) -> TaFuncResult:
    if resampling_strategy_args is None:
        resampling_strategy_args = {}

    evaluator = TrainEvaluator(
        backend=backend,
        metric=metric,
        configuration=config,
        splitter=splitter,
        splitter_kwargs=resampling_strategy_args,
        seed=seed,
        num_run=num_run,
        budget=budget,
        budget_type=budget_type,
    )
    result = evaluator.fit_predict_and_loss()
    return result


eval_holdout = partial(_eval, splitter=SingleWindowSplitter)

eval_sliding_window = partial(_eval, splitter=SlidingWindowSplitter)
