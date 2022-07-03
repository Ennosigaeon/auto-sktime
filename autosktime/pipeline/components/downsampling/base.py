from typing import Union

import numpy as np
import pandas as pd
from sktime.base._base import TagAliaserMixin
from sktime.datatypes import VectorizedDF

from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, HANDLES_PANEL, IGNORES_EXOGENOUS_X, \
    SUPPORTED_INDEX_TYPES
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimeTransformer, COMPONENT_PROPERTIES
from autosktime.pipeline.util import Int64Index


def fix_size(arr: np.array, original_size: int) -> np.ndarray:
    if arr.shape[0] > original_size:
        arrt = arr[:original_size]
    elif arr.shape[0] < original_size:
        arrt = np.hstack((arr, np.tile(arr[-1], original_size - arr.shape[0])))
    else:
        arrt = arr
    return arrt


class BaseDownSampling(AutoSktimeTransformer):
    _tags = {
        'capability:inverse_transform': True,
        'fit_is_empty': True,
        'y_inner_mtype': 'pd.DataFrame'
    }

    def _vectorize(self, methodname: str, **kwargs):
        """Vectorized/iterated loop over method of BaseTransformer.

        Uses transformers_ attribute to store one forecaster per loop index.
        """

        def unwrap(kwargs):
            """Unwrap kwargs to X, y, and reusable results of some method calls."""
            X = kwargs.pop("X")
            y = kwargs.pop("y", None)

            idx = X.get_iter_indices()
            n = len(idx)
            Xs = X.as_list()

            if y is None:
                ys = [None] * len(Xs)
            else:
                ys = y.as_list()

            return X, y, Xs, ys, n, idx

        FIT_METHODS = ["fit", "update"]
        TRAFO_METHODS = ["transform", "inverse_transform"]

        if methodname in FIT_METHODS:
            return self

        if methodname in TRAFO_METHODS:
            X, y, Xs, ys, n, _ = unwrap(kwargs)

            # fit/transform the i-th series/panel with a new clone of self
            Xts, yts = [], []
            for i in range(n):
                transformer = self.clone().fit(X=Xs[i], y=ys[i], **kwargs)
                method = getattr(transformer, methodname)
                res = method(X=Xs[i], y=ys[i], **kwargs)
                Xts += [res[0]]
                yts += [res[1]]
            Xt = X.reconstruct(Xts, overwrite_index=False)

            if y is not None:
                yt = y.reconstruct(yts, overwrite_index=False)
            else:
                yt = None

            return Xt, yt

    def transform(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame] = None):
        # check whether is fitted
        self.check_is_fitted()

        # input check and conversion for X/y
        X_inner, y_inner, metadata = self._check_X_y(X=X, y=y, return_metadata=True)

        if not isinstance(X_inner, VectorizedDF):
            Xt, yt = self._transform(X=X_inner, y=y_inner)
        else:
            # otherwise we call the vectorized version of predict
            Xt, yt = self._vectorize("transform", X=X_inner, y=y_inner)

        # convert to output mtype
        X_out = self._convert_output(Xt, metadata=metadata)
        if yt is not None:
            y_out = self._convert_output(yt, metadata=metadata)
        else:
            y_out = yt

        return X_out, y_out

    def _fit(self, X: Union[pd.Series, pd.DataFrame], y: pd.Series = None):
        return self

    def get_tags(self):
        tags = super(TagAliaserMixin, self).get_tags()
        return tags

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: True,
            HANDLES_PANEL: True,
            IGNORES_EXOGENOUS_X: True,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex, Int64Index]
        }
