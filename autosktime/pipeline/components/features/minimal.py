import numpy as np
from scipy.stats import kurtosis, skew

from autosktime.pipeline.components.features.base import BaseFeatureGenerator


class MinimalFeatureGenerator(BaseFeatureGenerator):

    def transform(self, X: np.ndarray) -> np.ndarray:
        features = [
            X.mean(axis=1),
            X.std(axis=1),
            np.zeros((X.shape[0], X.shape[2])),  # Gradient
            kurtosis(X, axis=1),
            skew(X, axis=1),
            np.zeros((X.shape[0], X.shape[2])),  # Crest
            X[:, -1, :]
        ]
        Xt = np.concatenate(features, axis=1)
        return Xt
