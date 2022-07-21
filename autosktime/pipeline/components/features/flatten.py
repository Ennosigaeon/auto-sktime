import numpy as np

from autosktime.pipeline.components.features.base import BaseFeatureGenerator


class FlatteningFeatureGenerator(BaseFeatureGenerator):

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xt = X.reshape(X.shape[0], -1)
        return Xt
