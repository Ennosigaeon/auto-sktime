import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils import weight_norm
from typing import Any

from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UniformFloatHyperparameter, EqualsCondition
from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, HANDLES_PANEL, IGNORES_EXOGENOUS_X, \
    SUPPORTED_INDEX_TYPES
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import COMPONENT_PROPERTIES, AutoSktimeComponent
from autosktime.pipeline.components.nn.network import BaseNetwork
from autosktime.pipeline.components.nn.network.head import LinearHead
from autosktime.pipeline.components.nn.util import NN_DATA
from autosktime.pipeline.util import Int64Index


class CnnBlock(nn.Module):
    def __init__(
            self,
            n_inputs: int,
            n_outputs: int,
            kernel_size: int,
            stride: int,
            dropout: float = 0.2
    ):
        super().__init__()
        self.net = nn.Sequential(
            weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size=kernel_size, stride=stride, padding='same')),
            nn.Dropout(dropout),
            nn.BatchNorm1d(n_outputs),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        return out


class CNN(BaseNetwork, AutoSktimeComponent):

    def __init__(
            self,
            num_layers: int = 3,
            num_filters: int = 16,
            kernel_size: int = 8,
            use_dropout: bool = True,
            dropout: float = 0.3,
            latent_size: int = 2,
            output_size: int = 1,
            random_state: np.random.RandomState = None
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.use_dropout = use_dropout
        self.dropout = dropout
        self.latent_size = latent_size
        self.output_size = output_size
        self.random_state = random_state

    def fit(self, data: NN_DATA, y: Any = None):
        self.num_features_ = data['X'].shape[1]

        layers = []
        for i in range(self.num_layers):
            in_channels = self.num_features_ if i == 0 else self.num_filters
            layers.append(
                CnnBlock(in_channels, self.num_filters, kernel_size=self.kernel_size, stride=1, dropout=self.dropout)
            )

        self.network_ = nn.Sequential(*layers)
        self.linear = nn.Linear(self.num_filters, self.latent_size)
        self.output_projector_ = LinearHead(self.latent_size, self.output_size,
                                         dropout=self.dropout if self.use_dropout else 0)

        return self

    def forward(self, x: torch.Tensor, device: torch.device, output_seq: bool = True) -> torch.Tensor:
        # swap sequence and feature dimensions for use with convolutional nets
        output = x.transpose(1, 2).contiguous()
        output = self.network_(output)
        output = output.transpose(1, 2).contiguous()
        output = self.linear(output)

        if not output_seq:
            output = self._get_last_seq_value(output)

        batch_size = x.shape[0]
        out = self.output_projector_(output).view(batch_size, -1)
        return out

    def _get_last_seq_value(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, -1:, :]

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: True,
            HANDLES_PANEL: True,
            IGNORES_EXOGENOUS_X: False,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex, Int64Index]
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        num_layers = UniformIntegerHyperparameter('num_layers', lower=1, upper=6, default_value=1)
        num_filters = UniformIntegerHyperparameter('num_filters', lower=4, upper=150, default_value=16, log=True)
        kernel_size = UniformIntegerHyperparameter('kernel_size', lower=2, upper=64, default_value=8, log=True)
        use_dropout = CategoricalHyperparameter('use_dropout', choices=[True, False], default_value=True)
        dropout = UniformFloatHyperparameter('dropout', lower=0, upper=0.5, default_value=0.3)

        cs = ConfigurationSpace()

        cs.add_hyperparameters([num_layers, num_filters, kernel_size, use_dropout, dropout])
        dropout_condition = EqualsCondition(dropout, use_dropout, True)
        cs.add_condition(dropout_condition)

        return cs
