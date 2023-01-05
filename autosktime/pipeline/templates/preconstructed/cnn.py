from torch import nn
from torch.nn.utils import weight_norm
from torch.optim.lr_scheduler import StepLR
from typing import Tuple, List, Any

from ConfigSpace import UnParametrizedHyperparameter, ConfigurationSpace
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimeComponent, UpdatablePipeline, SwappedInput
from autosktime.pipeline.components.data_preprocessing import VarianceThresholdComponent
from autosktime.pipeline.components.data_preprocessing.rescaling.minmax import MinMaxScalerComponent
from autosktime.pipeline.components.features.flatten import FlatteningFeatureGenerator
from autosktime.pipeline.components.nn.data_loader import ChunkedDataLoaderComponent
from autosktime.pipeline.components.nn.lr_scheduler import LearningRateScheduler
from autosktime.pipeline.components.nn.network.cnn import CNN
from autosktime.pipeline.components.nn.optimizer.optimizer import AdamOptimizer
from autosktime.pipeline.components.nn.trainer import TrainerComponent
from autosktime.pipeline.components.nn.util import DictionaryInput, NN_DATA
from autosktime.pipeline.templates import NNPanelRegressionPipeline
from autosktime.pipeline.templates.preconstructed import KMeansOperationCondition, FixedRecursivePanelReducer


class Fixed1dCNN(CNN):

    def fit(self, data: NN_DATA, y: Any = None):
        self.num_features_ = data['X'].shape[1]

        self.network_ = nn.Sequential(
            weight_norm(nn.Conv1d(self.num_features_, 10, kernel_size=10, stride=1, padding='same')),
            nn.BatchNorm1d(10),
            nn.Tanh(),

            weight_norm(nn.Conv1d(10, 10, kernel_size=10, stride=1, padding='same')),
            nn.BatchNorm1d(10),
            nn.Tanh(),

            weight_norm(nn.Conv1d(10, 10, kernel_size=10, stride=1, padding='same')),
            nn.BatchNorm1d(10),
            nn.Tanh(),

            weight_norm(nn.Conv1d(10, 10, kernel_size=10, stride=1, padding='same')),
            nn.BatchNorm1d(10),
            nn.Tanh(),

            weight_norm(nn.Conv1d(10, 3, kernel_size=10, stride=1, padding='same')),
            nn.BatchNorm1d(3),
            nn.Tanh(),
        )
        self.linear = nn.Sequential(nn.Dropout(0.5), nn.Linear(3, 100))
        self.output_projector_ = nn.Linear(100, self.output_size)

        return self

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


class FixedLearningRateScheduler(LearningRateScheduler):

    def fit(self, X: NN_DATA, y: Any = None) -> AutoSktimeComponent:
        optimizer = X['optimizer']
        self.scheduler = StepLR(optimizer, step_size=200, gamma=0.1)
        return self


class CNNRegressionPipeline(NNPanelRegressionPipeline):
    """
    Implementation of "Remaining Useful Life Estimation in Prognostics Using Deep Convolution Neural Networks"
    Source code adapted from https://github.com/jiaxiang-cheng/PyTorch-LSTM-for-RUL-Prediction
    Disable pynisher (use_pynisher) and multi-fidelity approximations (use_multi_fidelity) and set runcount_limit to 1
    when selecting this template.
    """

    def _get_pipeline_steps(self) -> List[Tuple[str, AutoSktimeComponent]]:
        pipeline = UpdatablePipeline(steps=[
            ('feature_generation', FlatteningFeatureGenerator(random_state=self.random_state)),
            ('variance_threshold', VarianceThresholdComponent(random_state=self.random_state)),
            ('dict', DictionaryInput()),
            ('data_loader', ChunkedDataLoaderComponent(random_state=self.random_state)),
            ('network', Fixed1dCNN(random_state=self.random_state)),
            ('optimizer', FixedAdamOptimizer(random_state=self.random_state)),
            ('lr_scheduler', FixedLearningRateScheduler()),
            ('trainer', TrainerComponent(iterations=250, use_timeout=False, random_state=self.random_state)),
        ])

        steps = [
            ('reduction', FixedRecursivePanelReducer(
                transformers=[
                    ('operation_condition', SwappedInput(KMeansOperationCondition(random_state=self.random_state))),
                    ('scaling', SwappedInput(MinMaxScalerComponent(feature_range=(-1, 1)))),
                ],
                estimator=pipeline,
                random_state=self.random_state,
                dataset_properties=self.dataset_properties)
             ),
        ]
        return steps
