from collections import Counter
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from sktime.performance_metrics.forecasting._classes import BaseForecastingErrorMetric

from autosktime.automl_common.common.ensemble_building.abstract_ensemble import AbstractEnsemble
from autosktime.automl_common.common.utils.backend import PIPELINE_IDENTIFIER_TYPE
from autosktime.constants import TASK_TYPES
from autosktime.metrics import calculate_loss
from autosktime.pipeline.templates.base import BasePipeline


class EnsembleSelection(AbstractEnsemble):

    def __init__(
            self,
            ensemble_size: int,
            task_type: int,
            metric: BaseForecastingErrorMetric,
            mode: str = 'fast',
            random_state: np.random.RandomState = None
    ) -> None:
        """ An ensemble of selected algorithms

        Fitting an EnsembleSelection generates an ensemble from the models
        generated during the search process. Can be further used for prediction.

        Parameters
        ----------
        task_type: int
            An identifier indicating which task is being performed.
        metric: Scorer
            The metric used to evaluate the models
        mode: str in ['fast', 'slow'] = 'fast'
            Which kind of ensemble generation to use
            *   'slow' - The original method used in Rich Caruana's ensemble selection.
            *   'fast' - A faster version of Rich Caruanas' ensemble selection.

        random_state: Optional[int | np.random.RandomState] = None
            The random_state used for ensemble selection.
            *   None - Uses numpy's default RandomState object
            *   int - Successive calls to fit will produce the same results
            *   RandomState - Truly random, each call to fit will produce different results, even with the same object.
        """
        self.ensemble_size = ensemble_size
        self.task_type = task_type
        self.metric = metric
        self.mode = mode
        self.random_state = random_state

    def __getstate__(self) -> Dict[str, Any]:
        # Cannot serialize a metric if it is user defined. That is, if doing pickle dump the metric won't be the same
        # as the one in __main__. we don't use the metric in the EnsembleSelection so this should be fine
        self.metric = None
        return self.__dict__

    def fit(
            self,
            predictions: List[pd.Series],
            labels: np.ndarray,
            identifiers: List[PIPELINE_IDENTIFIER_TYPE],
    ) -> AbstractEnsemble:
        self.ensemble_size = int(self.ensemble_size)
        if self.ensemble_size < 1:
            raise ValueError('Ensemble size cannot be less than one!')
        if self.task_type not in TASK_TYPES:
            raise ValueError(f'Unknown task type {self.task_type}.')
        if not isinstance(self.metric, BaseForecastingErrorMetric):
            raise ValueError(
                f'The provided metric must be an instance of {BaseForecastingErrorMetric}, nevertheless it is '
                f'{self.metric}({type(self.metric)})')
        if self.mode not in ('fast', 'slow'):
            raise ValueError(f'Unknown mode {self.mode}')

        self._fit(predictions, labels)
        self._calculate_weights()
        self.identifiers_ = identifiers
        return self

    def _fit(
            self,
            predictions: List[pd.Series],
            labels: np.ndarray,
    ) -> AbstractEnsemble:
        self.num_input_models_ = len(predictions)

        if self.mode == 'fast':
            func = self._fast
        else:
            func = self._slow
        self.indices_, self.trajectory_, self.train_loss_ = func(predictions, labels)

        return self

    def _fast(self, predictions: List[pd.Series], labels: pd.Series) -> Tuple[np.ndarray, np.ndarray, float]:
        """Fast version of Rich Caruana's ensemble selection method."""
        ensemble: List[pd.Series] = []
        trajectory = []
        order = []

        rand = check_random_state(self.random_state)

        weighted_ensemble_prediction = np.zeros(predictions[0].shape, dtype=np.float64)
        fant_ensemble_prediction = np.zeros(weighted_ensemble_prediction.shape, dtype=np.float64)

        for i in range(self.ensemble_size):
            losses = np.zeros((len(predictions)), dtype=np.float64)
            s = len(ensemble)
            if s > 0:
                np.add(weighted_ensemble_prediction, ensemble[-1], out=weighted_ensemble_prediction)

            # Memory-efficient averaging!
            for j, pred in enumerate(predictions):
                # fant_ensemble_prediction is the prediction of the current ensemble and should be
                # ([predictions[selected_prev_iterations] + predictions[j])/(s+1). We overwrite the contents of
                # fant_ensemble_prediction directly with weighted_ensemble_prediction + new_prediction and then scale
                # for avg
                np.add(weighted_ensemble_prediction, pred, out=fant_ensemble_prediction)
                np.multiply(fant_ensemble_prediction, (1. / float(s + 1)), out=fant_ensemble_prediction)

                losses[j] = calculate_loss(
                    solution=labels.values,
                    prediction=fant_ensemble_prediction,
                    task_type=self.task_type,
                    metric=self.metric
                )

            all_best = np.argwhere(losses == np.nanmin(losses)).flatten()

            best = rand.choice(all_best)

            ensemble.append(predictions[best])
            trajectory.append(losses[best])
            order.append(best)

            # Handle special case
            if len(predictions) == 1:
                break

        return np.array(order, dtype=np.int64), np.array(trajectory, dtype=np.float64), trajectory[-1]

    def _slow(self, predictions: List[pd.Series], labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Rich Caruana's ensemble selection method."""
        ensemble = []
        trajectory = []
        order = []

        for i in range(self.ensemble_size):
            losses = np.zeros([np.shape(predictions)[0]], dtype=np.float64)
            for j, pred in enumerate(predictions):
                ensemble.append(pred)
                ensemble_prediction = np.mean(np.array(ensemble), axis=0)
                losses[j] = calculate_loss(
                    solution=labels,
                    prediction=ensemble_prediction,
                    task_type=self.task_type,
                    metric=self.metric
                )
                ensemble.pop()
            best = np.nanargmin(losses)
            ensemble.append(predictions[best])
            trajectory.append(losses[best])
            order.append(best)

            # Handle special case
            if len(predictions) == 1:
                break

        return np.array(order, dtype=np.int64), np.array(trajectory, dtype=np.float64), trajectory[-1]

    def _calculate_weights(self) -> None:
        ensemble_members = Counter(self.indices_).most_common()
        weights = np.zeros((self.num_input_models_,), dtype=np.float64)
        for ensemble_member in ensemble_members:
            weight = float(ensemble_member[1]) / self.ensemble_size
            weights[ensemble_member[0]] = weight

        if np.sum(weights) < 1:
            weights = weights / np.sum(weights)

        self.weights_ = weights

    def __str__(self) -> str:
        trajectory_str = ' '.join([f'{id_}: {perf:.5f}' for id_, perf in enumerate(self.trajectory_)])
        identifiers_str = ' '.join([f'{id_}' for idx, id_ in enumerate(self.identifiers_) if self.weights_[idx] > 0])
        return ('Ensemble Selection:\n'
                f'\tTrajectory: {trajectory_str}\n'
                f'\tMembers: {self.indices_}\n'
                f'\tWeights: {self.weights_}\n'
                f'\tIdentifiers: {identifiers_str}\n')

    def get_models_with_weights(
            self,
            models: Dict[PIPELINE_IDENTIFIER_TYPE, BasePipeline]
    ) -> List[Tuple[float, BasePipeline]]:
        output = []
        for i, weight in enumerate(self.weights_):
            if weight > 0.0:
                identifier = self.identifiers_[i]
                model = models[identifier]
                output.append((weight, model))

        output.sort(reverse=True, key=lambda t: t[0])

        return output

    def get_selected_model_identifiers(self) -> List[PIPELINE_IDENTIFIER_TYPE]:
        output = []

        for i, weight in enumerate(self.weights_):
            identifier = self.identifiers_[i]
            if weight > 0.0:
                output.append(identifier)

        return output

    def predict(self, base_models_predictions: List[Union[pd.DataFrame, pd.Series]]) -> pd.DataFrame:
        predictions = np.array(base_models_predictions)
        pred = np.average(predictions[self.indices_], weights=self.weights_[self.indices_], axis=0)
        return pd.DataFrame(pred, columns=base_models_predictions[0].columns, index=base_models_predictions[0].index)

    def get_validation_performance(self) -> float:
        return self.trajectory_[-1]
