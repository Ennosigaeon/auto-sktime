import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from typing import Any

from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UniformFloatHyperparameter, EqualsCondition
from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, HANDLES_PANEL, IGNORES_EXOGENOUS_X, \
    SUPPORTED_INDEX_TYPES
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimeComponent, COMPONENT_PROPERTIES
from autosktime.pipeline.components.nn.network.base import BaseNetwork
from autosktime.pipeline.components.nn.network.head import LinearHead
from autosktime.pipeline.components.nn.util import NN_DATA, Chomp1d
from autosktime.pipeline.util import Int64Index


class TemporalBlock(nn.Module):
    def __init__(
            self,
            n_inputs: int,
            n_outputs: int,
            kernel_size: int,
            stride: int,
            dilation: int,
            padding: int,
            dropout: float = 0.2
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNetwork(BaseNetwork, AutoSktimeComponent):

    def __init__(
            self,
            num_layers: int = 3,
            num_filters: int = 16,
            kernel_size: int = 8,
            use_dropout: bool = True,
            dropout: float = 0.3,
            output_size: int = 1,
            random_state: np.random.RandomState = None
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.use_dropout = use_dropout
        self.dropout = dropout
        self.output_size = output_size
        self.random_state = random_state

    def fit(self, data: NN_DATA, y: Any = None):
        self.num_features_ = data['X'].shape[1]

        layers = []
        for i in range(self.num_layers):
            dilation_size = 2 ** i
            in_channels = self.num_features_ if i == 0 else self.num_filters
            out_channels = self.num_filters
            layers += [
                TemporalBlock(in_channels, out_channels, kernel_size=self.kernel_size, stride=1, dilation=dilation_size,
                              padding=(self.kernel_size - 1) * dilation_size,
                              dropout=self.dropout if self.use_dropout else 0)
            ]

        self.network_ = nn.Sequential(*layers)
        self.output_projector_ = LinearHead(self.num_filters, self.output_size,
                                            dropout=self.dropout if self.use_dropout else 0)

        return self

    def forward(self, x: torch.Tensor, device: torch.device, output_seq: bool = True) -> torch.Tensor:
        # swap sequence and feature dimensions for use with convolutional nets
        output = x.transpose(1, 2).contiguous()
        output = self.network_(output)
        output = output.transpose(1, 2).contiguous()

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
        num_layers = UniformIntegerHyperparameter('num_layers', lower=1, upper=4, default_value=1)
        num_filters = UniformIntegerHyperparameter('num_filters', lower=4, upper=128, default_value=16, log=True)
        kernel_size = UniformIntegerHyperparameter('kernel_size', lower=2, upper=64, default_value=8, log=True)
        use_dropout = CategoricalHyperparameter('use_dropout', choices=[True, False], default_value=True)
        dropout = UniformFloatHyperparameter('dropout', lower=0, upper=0.5, default_value=0.1)

        cs = ConfigurationSpace()

        cs.add_hyperparameters([num_layers, num_filters, kernel_size, use_dropout, dropout])
        dropout_condition = EqualsCondition(dropout, use_dropout, True)
        cs.add_condition(dropout_condition)

        return cs
