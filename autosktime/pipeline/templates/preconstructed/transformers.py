import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, List, Tuple

from ConfigSpace import ConfigurationSpace, UnParametrizedHyperparameter
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimeComponent, COMPONENT_PROPERTIES, UpdatablePipeline, \
    SwappedInput
from autosktime.pipeline.components.data_preprocessing import VarianceThresholdComponent
from autosktime.pipeline.components.data_preprocessing.rescaling.standardize import StandardScalerComponent
from autosktime.pipeline.components.features.flatten import FlatteningFeatureGenerator
from autosktime.pipeline.components.index import AddIndexComponent
from autosktime.pipeline.components.nn.data_loader import ChunkedDataLoaderComponent
from autosktime.pipeline.components.nn.lr_scheduler import LearningRateScheduler
from autosktime.pipeline.components.nn.network import BaseNetwork
from autosktime.pipeline.components.nn.optimizer.optimizer import AdamOptimizer
from autosktime.pipeline.components.nn.trainer import TrainerComponent
from autosktime.pipeline.components.nn.util import NN_DATA, DictionaryInput
from autosktime.pipeline.components.normalizer.standardize import TargetStandardizeComponent
from autosktime.pipeline.templates import NNPanelRegressionPipeline
from autosktime.pipeline.templates.preconstructed import FixedRecursivePanelReducer


class Transformer(BaseNetwork, AutoSktimeComponent):

    def __init__(self, latent_size: int = 128, n_layers: int = 2, n_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.latent_size = latent_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout

    def fit(self, data: NN_DATA, y: Any = None):
        self.num_features_ = data['X'].shape[1]
        device = data['device']

        self.network_ = nn.Sequential(
            LocalFeatureExtraction(self.num_features_, self.latent_size, device),
            Encoder(self.latent_size, self.n_layers, self.n_heads, self.dropout)
        )
        self.output_projector_ = nn.Linear(self.latent_size, 1)

        return self

    def forward(self, x: torch.Tensor, device: torch.device, output_seq: bool = True) -> torch.Tensor:
        latent = self.network_(x)
        output = self.output_projector_(latent)
        return output[:, :, -1]

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        pass

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        return ConfigurationSpace()


class LocalFeatureExtraction(nn.Module):

    def __init__(self, n_features: int, hidden_size: int, device: str):
        super().__init__()
        self.n_features = n_features
        self.gcu = Gating(n_features, hidden_size, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = x[:, None, :, :]

        h_i = self.gcu(xt)
        out = h_i.reshape(h_i.shape[0], 1, h_i.shape[-1])
        return out


class Gating(nn.Module):

    def __init__(self, n_features: int, hidden_size: int, device: str):
        super().__init__()
        self.n_features = n_features

        # reset gate r_i
        self.W_r = nn.Parameter(torch.zeros(n_features, n_features).to(device))
        self.V_r = nn.Parameter(torch.zeros(n_features, n_features).to(device))
        self.b_r = nn.Parameter(torch.zeros(n_features).to(device))

        # update gate u_i
        self.W_u = nn.Parameter(torch.zeros(n_features, n_features).to(device))
        self.V_u = nn.Parameter(torch.zeros(n_features, n_features).to(device))
        self.b_u = nn.Parameter(torch.zeros(n_features).to(device))

        stdv = 1.0 / math.sqrt(n_features)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

        self.cnn_layer = nn.Conv2d(1, 1, kernel_size=(3, 1), stride=1)
        self.linear_layer = nn.Linear(n_features, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shape becomes (batch_size, 1, 1, n_features) as the nn.conv2d has output channel as 1 but the convolution is
        # applied on whole window_size
        h_i = self.cnn_layer(x)

        # only applying the gating on the current row, i.e. the center value of x
        center = x.shape[2] // 2
        x_i = x[:, :, center:(center + 1), :]

        r_i = torch.sigmoid(torch.matmul(h_i, self.W_r) + torch.matmul(x_i, self.V_r) + self.b_r)
        u_i = torch.sigmoid(torch.matmul(h_i, self.W_u) + torch.matmul(x_i, self.V_u) + self.b_u)

        # the output of the gating mechanism
        hh_i = torch.mul(h_i, u_i) + torch.mul(x_i, r_i)

        return self.linear_layer(hh_i)


class Encoder(nn.Module):
    def __init__(self, n_features: int, n_layers: int, n_heads: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(n_features),
            *[EncoderLayer(n_features, n_heads, dropout) for _ in range(n_layers)]
        )

    def forward(self, x):
        return self.network(x)


class EncoderLayer(nn.Module):
    def __init__(self, n_features: int, n_heads: int, dropout: float = 0.5):
        super().__init__()
        self.norm_1 = nn.LayerNorm(n_features)
        self.norm_2 = nn.LayerNorm(n_features)
        self.attn = MultiHeadAttention(n_features, n_heads, dropout)
        self.ff = FeedForward(n_features)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, n_features: int, n_heads: int, dropout: float = 0.5):
        super().__init__()

        self.d_model = n_features
        self.d_k = n_features // n_heads
        self.h = n_heads

        self.q_linear = nn.Linear(n_features, n_features)
        self.v_linear = nn.Linear(n_features, n_features)
        self.k_linear = nn.Linear(n_features, n_features)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(n_features, n_features)

    def forward(self, q, k, v):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = self.attention(q, k, v)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)

        output = self.out(concat)

        return output

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        output = torch.matmul(scores, v)
        return output


class FeedForward(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 512, dropout: float = 0.5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class FixedAdamOptimizer(AdamOptimizer):

    def get_hyperparameter_search_space(self, dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        lr = UnParametrizedHyperparameter('lr', 1e-3)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([lr])
        return cs


class TransformerRegressionPipeline(NNPanelRegressionPipeline):
    """
    Implementation of "Remaining useful life estimation via transformer encoder enhanced by a gated convolutional unit"
    Source code adapted from https://github.com/jiaxiang-cheng/PyTorch-Transformer-for-RUL-Prediction
    Disable pynisher (use_pynisher) and multi-fidelity approximations (use_multi_fidelity) when selecting this template.
    Furthermore, use a short timeout as the config space contains only a single configuration (see https://github.com/automl/SMAC3/issues/21)
    """

    def _get_pipeline_steps(self) -> List[Tuple[str, AutoSktimeComponent]]:
        pipeline = UpdatablePipeline(steps=[
            ('feature_generation', FlatteningFeatureGenerator(random_state=self.random_state)),
            ('variance_threshold', VarianceThresholdComponent(random_state=self.random_state)),
            ('scaling', StandardScalerComponent(random_state=self.random_state)),
            ('dict', DictionaryInput()),
            ('data_loader', ChunkedDataLoaderComponent(window_length=3, random_state=self.random_state)),
            ('network', Transformer()),
            ('optimizer', FixedAdamOptimizer(random_state=self.random_state)),
            ('lr_scheduler', LearningRateScheduler()),
            ('trainer', TrainerComponent(use_timeout=False, random_state=self.random_state)),
        ])

        steps = [
            ('scaling', TargetStandardizeComponent(random_state=self.random_state)),
            ('reduction', FixedRecursivePanelReducer(
                transformers=[
                    ('add_index', SwappedInput(AddIndexComponent())),
                ],
                estimator=pipeline,
                random_state=self.random_state,
                dataset_properties=self.dataset_properties)
             ),
        ]
        return steps
