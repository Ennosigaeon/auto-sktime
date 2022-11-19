import numpy as np
import pandas as pd
import torch
from torch import nn
from typing import Any

from ConfigSpace import ConfigurationSpace, GreaterThanCondition, AndConjunction, EqualsCondition, \
    CategoricalHyperparameter, UniformIntegerHyperparameter, UniformFloatHyperparameter
from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, HANDLES_PANEL, IGNORES_EXOGENOUS_X, \
    SUPPORTED_INDEX_TYPES
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimeComponent
from autosktime.pipeline.components.nn.network.base import BaseNetwork
from autosktime.pipeline.components.nn.network.head import LinearHead
from autosktime.pipeline.components.nn.util import NN_DATA
from autosktime.pipeline.util import Int64Index


class RecurrentNetwork(BaseNetwork, AutoSktimeComponent):

    def __init__(
            self,
            cell_type: str = 'lstm',
            hidden_size: int = 64,
            num_layers: int = 1,
            use_dropout: bool = True,
            dropout: float = 0.3,
            latent_size: int = 2,
            output_size: int = 1,
            random_state: np.random.RandomState = None
    ):
        super().__init__()
        self.cell_type = cell_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_dropout = use_dropout
        self.dropout = dropout
        self.latent_size = latent_size
        self.output_size = output_size
        self.random_state = random_state

    def fit(self, data: NN_DATA, y: Any = None):
        if self.cell_type == 'lstm':
            rnn_class = nn.LSTM
        elif self.cell_type == 'gru':
            rnn_class = nn.GRU
        else:
            raise ValueError(f'RNN type {self.cell_type} is not supported. supported: [lstm, gru]')

        self.num_features_ = data['X'].shape[1]

        self.network_ = rnn_class(
            input_size=self.num_features_,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.use_dropout and self.num_layers > 1 else 0,
            bidirectional=False,
            batch_first=True,
        )
        self.linear = nn.Linear(self.hidden_size, self.latent_size)
        self.output_projector_ = LinearHead(self.latent_size, self.output_size)

        return self

    def forward(self, x: torch.Tensor, device: torch.device, output_seq: bool = True) -> torch.Tensor:
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_().to(device)

        if self.cell_type == 'lstm':
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_().to(device)
            output, hn = self.network_(x, (h0, c0))
        else:
            output, hn = self.network_(x, h0)

        if not output_seq:
            output = self._get_last_seq_value(output)

        output = self.linear(output)
        out = self.output_projector_(output).view(batch_size, -1)
        return out

    def _get_last_seq_value(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, -1:, :]

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None):
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: True,
            HANDLES_PANEL: True,
            IGNORES_EXOGENOUS_X: False,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex, Int64Index]
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        cell_type = CategoricalHyperparameter('cell_type', ['lstm', 'gru'], default_value='lstm')
        num_layers = UniformIntegerHyperparameter('num_layers', lower=1, upper=3, default_value=1)
        hidden_size = UniformIntegerHyperparameter('hidden_size', lower=16, upper=512, default_value=100, log=True)
        use_dropout = CategoricalHyperparameter('use_dropout', [True, False], default_value=True)
        dropout = UniformFloatHyperparameter('dropout', lower=0., upper=0.5, default_value=0.3)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([cell_type, num_layers, hidden_size, use_dropout, dropout])

        cs.add_condition(AndConjunction(
            EqualsCondition(dropout, use_dropout, True), GreaterThanCondition(dropout, num_layers, 1))
        )

        return cs
