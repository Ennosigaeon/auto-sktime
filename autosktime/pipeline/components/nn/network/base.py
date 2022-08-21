from typing import Optional

import numpy as np
from torch import nn

from autosktime.pipeline.components.nn.util import NN_DATA


class BaseNetwork(nn.Module):

    def __init__(self, random_state: np.random.RandomState = None):
        super().__init__()
        self.random_state = random_state

        self.num_features_: Optional[int] = None
        self.network_: Optional[nn.Module] = None
        self.output_projector_: Optional[nn.Module] = None

    def transform(self, X: NN_DATA) -> NN_DATA:
        X.update({'network': self})
        return X
