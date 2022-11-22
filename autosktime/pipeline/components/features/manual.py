import warnings

import numpy as np
from scipy.stats import kurtosis, skew

from autosktime.pipeline.components.features.base import BaseFeatureGenerator


class ManualFeatureGenerator(BaseFeatureGenerator):

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] == 1:
            gradient = np.zeros((X.shape[0], X.shape[2]))
        else:
            gradient = np.gradient(X, axis=1).sum(axis=1) / X.shape[1]

        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=RuntimeWarning)
            features = [
                X.mean(axis=1)[:, 0:1],
                X.std(axis=1)[:, 0:1],
                gradient[:, 0:1],
                kurtosis(X, axis=1)[:, 0:1],
                skew(X, axis=1)[:, 0:1],
                self._crest(X)[:, 0:1],
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
