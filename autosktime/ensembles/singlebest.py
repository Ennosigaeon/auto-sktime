import os
from typing import List, Tuple, Union, Dict

import numpy as np
from smac.runhistory.runhistory import RunHistory

from autosktime.automl_common.common.ensemble_building.abstract_ensemble import AbstractEnsemble
from autosktime.automl_common.common.utils.backend import Backend, PIPELINE_IDENTIFIER_TYPE
from autosktime.metrics import BaseMetric, get_cost_of_crash
from autosktime.pipeline.templates.base import BasePipeline


class SingleBest(AbstractEnsemble):
    """
    In the case of a crash, this class searches for the best individual model.

    Such model is returned as an ensemble of a single object, to comply with the expected interface of an
    AbstractEnsemble.
    """

    def __init__(
            self,
            metric: BaseMetric,
            run_history: RunHistory,
            seed: int,
            backend: Backend,
    ):
        self.metric = metric
        self.seed = seed
        self.backend = backend

        # Add some default values -- at least 1 model in ensemble is assumed
        self.indices_ = [0]
        self.weights_ = [1.0]
        self.run_history = run_history
        self.identifiers_ = [self.get_incumbent()]

    def get_incumbent(self) -> PIPELINE_IDENTIFIER_TYPE:
        """
        This method parses the run history, to identify the best performing model.
        """
        best_model_identifier = None
        best_model_score = get_cost_of_crash(self.metric)

        for run_key in self.run_history.data.keys():
            run_value = self.run_history.data[run_key]

            if run_value.cost < best_model_score:
                # Make sure that the individual best model actually exists
                model_dir = self.backend.get_numrun_directory(self.seed, run_key.config_id, run_key.budget)
                model_file_name = self.backend.get_model_filename(self.seed, run_key.config_id, run_key.budget)
                file_path = os.path.join(model_dir, model_file_name)
                if not os.path.exists(file_path):
                    continue

                best_model_identifier = (self.seed, run_value.additional_info['num_run'], run_key.budget)
                best_model_score = run_value.cost

        if not best_model_identifier:
            raise ValueError(
                'No valid model found in run history. This means smac was not able to fit a valid model. Please check '
                'the log file for errors.'
            )

        return best_model_identifier

    def __str__(self) -> str:
        return 'Single Model Selection:\n\tMembers: %s' \
               '\n\tWeights: %s\n\tIdentifiers: %s' % \
               (self.indices_, self.weights_,
                ' '.join([str(identifier) for idx, identifier in
                          enumerate(self.identifiers_)
                          if self.weights_[idx] > 0]))

    def get_models_with_weights(
            self,
            models: Dict[PIPELINE_IDENTIFIER_TYPE, BasePipeline]
    ) -> List[Tuple[float, BasePipeline]]:
        return [(self.weights_[0], models[self.identifiers_[0]])]

    def get_selected_model_identifiers(self) -> List[PIPELINE_IDENTIFIER_TYPE]:
        return self.identifiers_

    def predict(self, predictions: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        raise NotImplementedError('Ensemble should always construct a sktime.forecasting.compose.EnsembleForecaster '
                                  'that is used for actual predictions.')