import torch
from torch import nn


class NetworkHead(nn.Module):
    network_: nn.Module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network_(x)


class LinearHead(NetworkHead):

    def __init__(self, input_size: int, output_size: int = 1, dropout: float = 0.):
        super().__init__()
        self.network_ = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, output_size),
        )


class GAPHead(NetworkHead):

    def __init__(self, output_size: int = 1, dropout: float = 0.):
        super().__init__()
        self.network_ = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size),
            nn.Dropout(dropout),
            nn.Linear(1, output_size)
        )
