import numpy as np
from scipy.stats import kurtosis, skew

from autosktime.pipeline.components.features.base import BaseFeatureGenerator


class MinimalFeatureGenerator(BaseFeatureGenerator):

    def transform(self, X: np.ndarray) -> np.ndarray:
        features = [
            X.mean(axis=1),
            X.std(axis=1),
            np.gradient(X, axis=1).sum(axis=1) / X.shape[1],
            kurtosis(X, axis=1),
            skew(X, axis=1),
            self._crest(X),
            X[:, -1, :]
        ]
        Xt = np.concatenate(features, axis=1)
        return Xt

    @staticmethod
    def _crest(x: np.ndarray, pctarr: float = 100.0, intep: str = 'midpoint'):
        peak = np.percentile(np.abs(x), pctarr, method=intep, axis=1)
        sig = x.std(axis=1)

        CF = peak / sig
        np.nan_to_num(CF, posinf=0)
        return CF
