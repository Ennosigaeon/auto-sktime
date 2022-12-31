import torch
from torch import nn
from typing import Tuple, List, Any

from ConfigSpace import CategoricalHyperparameter, UnParametrizedHyperparameter, ConfigurationSpace
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimeComponent, UpdatablePipeline, SwappedInput
from autosktime.pipeline.components.data_preprocessing import VarianceThresholdComponent
from autosktime.pipeline.components.features.flatten import FlatteningFeatureGenerator
from autosktime.pipeline.components.nn.data_loader import ChunkedDataLoaderComponent
from autosktime.pipeline.components.nn.lr_scheduler import LearningRateScheduler
from autosktime.pipeline.components.nn.network.rnn import RecurrentNetwork
from autosktime.pipeline.components.nn.optimizer.optimizer import AdamOptimizer
from autosktime.pipeline.components.nn.trainer import TrainerComponent
from autosktime.pipeline.components.nn.util import DictionaryInput, NN_DATA
from autosktime.pipeline.components.reduction.panel import RecursivePanelReducer
from autosktime.pipeline.templates import NNPanelRegressionPipeline
from autosktime.pipeline.templates.base import get_pipeline_search_space
from autosktime.pipeline.templates.preconstructed import KMeansOperationCondition, DataScaler


class FixedRecursivePanelReducer(RecursivePanelReducer):

    def get_hyperparameter_search_space(self, dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        window_length = CategoricalHyperparameter('window_length', [1])
        step_size = UnParametrizedHyperparameter('step_size', 0.001)

        estimator = get_pipeline_search_space(self.estimator.steps, dataset_properties=dataset_properties)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([window_length, step_size])
        cs.add_configuration_space('estimator', estimator)
        return cs


class FixedLSTM(RecurrentNetwork):

    def fit(self, data: NN_DATA, y: Any = None):
        self.num_features_ = data['X'].shape[1]

        self.lstm_1 = nn.LSTM(input_size=self.num_features_, hidden_size=32, num_layers=1, batch_first=True,
                              dropout=0.1)
        self.lstm_2 = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True, dropout=0.1)
        self.output_projector_ = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(8, self.output_size)
        )

        return self

    def forward(self, x: torch.Tensor, device: torch.device, output_seq: bool = True) -> torch.Tensor:
        h_0 = torch.zeros(1, x.size(0), 32).to(device)
        c_0 = torch.zeros(1, x.size(0), 32).to(device)
        output, _ = self.lstm_1(x, (h_0, c_0))

        h_1 = torch.zeros(1, x.size(0), 64).to(device)
        c_1 = torch.zeros(1, x.size(0), 64).to(device)
        output, (hn_2, cn) = self.lstm_2(output, (h_1, c_1))

        hn_o = hn_2.view(-1, 64)
        out = self.output_projector_(hn_o)
        return out

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        return cs


class FixedAdamOptimizer(AdamOptimizer):

    def get_hyperparameter_search_space(self, dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        lr = UnParametrizedHyperparameter('lr', 1e-3)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([lr])
        return cs


class LSTMRegressionPipeline(NNPanelRegressionPipeline):
    """
    Implementation of "Long Short-Term Memory Network for Remaining Useful Life estimation"
    Source code adapted from https://github.com/jiaxiang-cheng/PyTorch-LSTM-for-RUL-Prediction
    Disable pynisher (use_pynisher) and multi-fidelity approximations (use_multi_fidelity) when selecting this template.
    Furthermore, use a short timeout as the config space contains only a single configuration (see https://github.com/automl/SMAC3/issues/21)
    """

    def _get_pipeline_steps(self) -> List[Tuple[str, AutoSktimeComponent]]:
        pipeline = UpdatablePipeline(steps=[
            ('feature_generation', FlatteningFeatureGenerator(random_state=self.random_state)),
            ('variance_threshold', VarianceThresholdComponent(random_state=self.random_state)),
            ('dict', DictionaryInput()),
            ('data_loader', ChunkedDataLoaderComponent(random_state=self.random_state)),
            ('network', FixedLSTM(random_state=self.random_state)),
            ('optimizer', FixedAdamOptimizer(random_state=self.random_state)),
            ('lr_scheduler', LearningRateScheduler()),
            ('trainer', TrainerComponent(use_timeout=False, random_state=self.random_state)),
        ])

        steps = [
            ('reduction', FixedRecursivePanelReducer(
                transformers=[
                    ('operation_condition', SwappedInput(KMeansOperationCondition(random_state=self.random_state))),
                    ('scaling', SwappedInput(DataScaler())),
                ],
                estimator=pipeline,
                random_state=self.random_state,
                dataset_properties=self.dataset_properties)
             ),
        ]
        return steps
