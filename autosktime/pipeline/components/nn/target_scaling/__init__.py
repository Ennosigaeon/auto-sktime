from typing import Union

import pandas as pd
from sklearn.exceptions import NotFittedError

from autosktime.pipeline.components.data_preprocessing import RescalingChoice


class TargetScalerChoice(RescalingChoice):

    def _fit(self, X: Union[pd.Series, pd.DataFrame], y: pd.Series = None):
        if self.estimator is None:
            raise NotFittedError()

        # noinspection PyUnresolvedReferences
        self.estimator.fit(y)
        return self

    def _transform(self, X: Union[pd.Series, pd.DataFrame], y: pd.DataFrame = None):
        if self.estimator is None:
            raise NotFittedError()

        # noinspection PyUnresolvedReferences
        yt = self.estimator.transform(y)

        return X, yt
