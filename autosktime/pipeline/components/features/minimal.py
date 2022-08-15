import numpy as np
from scipy.stats import kurtosis, skew

from autosktime.pipeline.components.features.base import BaseFeatureGenerator


class MinimalFeatureGenerator(BaseFeatureGenerator):

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_feat = self._get_features(X)
        if X.shape[1] == 1:
            gradient = np.zeros((X.shape[0], X.shape[2]))
        else:
            gradient = np.gradient(X_feat, axis=1).sum(axis=1) / X_feat.shape[1]

        features = [
            X_feat.mean(axis=1),
            X_feat.std(axis=1),
            gradient,
            kurtosis(X_feat, axis=1),
            skew(X_feat, axis=1),
            self._crest(X_feat),
            X[:, -1, :]
        ]
        Xt = np.concatenate(features, axis=1)
        return Xt

    @staticmethod
    def _crest(x: np.ndarray, pctarr: float = 100.0, intep: str = 'midpoint'):
        peak = np.percentile(np.abs(x), pctarr, method=intep, axis=1)
        sig = x.std(axis=1)

        CF = np.divide(peak, sig, out=np.zeros_like(peak), where=sig != 0)
        return CF
